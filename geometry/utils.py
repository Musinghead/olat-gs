"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import random
from typing import Optional, Sequence

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import torch
import torch.nn.functional as F
from datasets.utils import Rays, namedtuple_map
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfacc.estimators.prop_net import PropNetEstimator
from nerfacc.grid import ray_aabb_intersect, traverse_grids
from nerfacc.volrend import (
    accumulate_along_rays_,
    accumulate_along_rays,
    render_weight_from_density,
    render_weight_from_alpha,
    rendering,
)

import mcubes

from radiance_fields.field import SDFNetwork, RenderingNetwork, SingleVarianceNetwork, RenderingNetworkPl


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).cuda().split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).cuda().split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).cuda().split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def render_rays_with_occgrid_neus_olat(
    # scene
    sdf_net: SDFNetwork,
    render_net: RenderingNetworkPl,
    variance_net: SingleVarianceNetwork,

    estimator: OccGridEstimator,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    pl_pos: torch.Tensor,
    cos_anneal_ratio: float = 1.0,
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    is_training: bool = False,
    render_norm: bool = False
):
    """render a batch of rays, used during training"""
    def alpha_fn(t_starts, t_ends, ray_indices):
        if t_starts.shape[0] == 0:
            alphas = torch.empty((0, 1), device=t_starts.device)
        else:
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = (
                t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            )
            t_dists = t_ends - t_starts

            gradients, sdf, _ = sdf_net.gradient(positions)

            inv_s = variance_net(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)

            true_cos = (t_dirs * gradients).sum(-1, keepdim=True)

            # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
            # the cos value "not dead" at the beginning training iterations, for better convergence.
            iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                        F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

            # Estimate signed distances at section points
            estimated_next_sdf = sdf + iter_cos * t_dists.reshape(-1, 1) * 0.5
            estimated_prev_sdf = sdf - iter_cos * t_dists.reshape(-1, 1) * 0.5

            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alphas = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        return alphas.squeeze(-1)

    # freeze sdf net and s_val during sampling, reduce computing grad
    for param in sdf_net.parameters():
        param.requires_grad_(False)
    
    for param in variance_net.parameters():
        param.requires_grad_(False)

    # sampling is set torch.no_grad()
    ray_indices, t_starts, t_ends = estimator.sampling(
        rays_o,
        rays_d,
        alpha_fn=alpha_fn,
        near_plane=near_plane,
        far_plane=far_plane,
        render_step_size=render_step_size,
        stratified=sdf_net.training,
        cone_angle=cone_angle,
        alpha_thre=alpha_thre,
    )

    for param in sdf_net.parameters():
        param.requires_grad_(True)
    
    for param in variance_net.parameters():
        param.requires_grad_(True)

    t_origins = rays_o[ray_indices]
    t_dirs = rays_d[ray_indices]
    t_pl_pos = pl_pos[ray_indices]
    positions = (
                t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            )
    t_dists = t_ends - t_starts

    gradients, sdf, feature_vector = sdf_net.gradient(positions)

    inv_s = variance_net(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
    inv_s = inv_s.expand(sdf.shape[0], 1)

    true_cos = (t_dirs * gradients).sum(-1, keepdim=True)

    # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
    # the cos value "not dead" at the beginning training iterations, for better convergence.
    iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

    # Estimate signed distances at section points
    estimated_next_sdf = sdf + iter_cos * t_dists.reshape(-1, 1) * 0.5
    estimated_prev_sdf = sdf - iter_cos * t_dists.reshape(-1, 1) * 0.5

    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf
    # (n, 1)
    alphas = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

    # normalized_grad = F.normalize(gradients, p=2, dim=-1)
    # rgbs = render_net(positions, normalized_grad, t_dirs, t_pl_pos, feature_vector)
    rgbs = render_net(positions, gradients, t_dirs, t_pl_pos, feature_vector)

    n_rays = rays_o.shape[0]
    weights, trans = render_weight_from_alpha(
        alphas.reshape(-1),
        ray_indices=ray_indices,
        n_rays=n_rays
    )

    colors = accumulate_along_rays(
        weights, values=rgbs, ray_indices=ray_indices, n_rays=n_rays
    )
    opacities = accumulate_along_rays(
        weights, values=None, ray_indices=ray_indices, n_rays=n_rays
    )

    # omit rendering normal image during training
    normals = None
    if render_norm:
        normals = accumulate_along_rays(
            weights, values=gradients, ray_indices=ray_indices, n_rays=n_rays
        )
    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    # gradients regularization
    pts_norm = torch.linalg.norm(positions, ord=2, dim=-1, keepdim=True).reshape(-1)
    # inside_sphere = (pts_norm < 1.0).float().detach()
    relax_inside_sphere = (pts_norm < 1.2).float().detach()
    
    # Eikonal loss
    gradient_error = (torch.linalg.norm(gradients.reshape(-1, 3), ord=2,
                                        dim=-1) - 1.0) ** 2
    gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

    return colors, normals, len(positions), gradient_error, inv_s.mean(), opacities

# neus version, render with alpha, also return gradients to be regularized
def render_image_with_occgrid_neus_olat(
    # scene
    sdf_net: SDFNetwork,
    render_net: RenderingNetworkPl,
    variance_net: SingleVarianceNetwork,

    estimator: OccGridEstimator,
    rays: Rays,
    pl_pos: torch.Tensor,
    # rendering options
    cos_anneal_ratio: float = 1.0,
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
        pl_pos = pl_pos.reshape(rays.origins.shape[0], -1)
    else:
        num_rays, _ = rays_shape

    results = []
    chunk = test_chunk_size

    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)

        rays_o = chunk_rays.origins
        rays_d = chunk_rays.viewdirs
        pl_pos_chunk = pl_pos[i : i + chunk]

        colors, normals, _, _, _, _ = render_rays_with_occgrid_neus_olat(
            sdf_net=sdf_net,
            render_net=render_net,
            variance_net=variance_net,
            estimator=estimator,
            rays_o=rays_o,
            rays_d=rays_d,
            pl_pos=pl_pos_chunk,
            cos_anneal_ratio=cos_anneal_ratio,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
            render_norm=True
        )

        chunk_results = [colors.detach(), normals.detach()]
        results.append(chunk_results)

    batch_colors, batch_normals = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        batch_colors.view((*rays_shape[:-1], -1)),
        batch_normals.view((*rays_shape[:-1], -1))
    )
