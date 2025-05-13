#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene import Scene
from utils.sh_utils import eval_sh, RGB2SH
from utils.general_utils import quat_mul, build_rotation, dot_prod
from utils.graphics_utils import getWorld2ViewQuat
from scene.cameras import Camera
import numpy as np


def linear2srgb(f):
    return torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055)


def query_olat(viewpoint_camera: Camera, pc: GaussianModel, lgt_depth):
    """
    query olat values without rasterization

    """

    means3D = pc.get_xyz

    refine_means3D = means3D

    # calculate wi&wo
    wi = viewpoint_camera.pl_pos - means3D.detach()
    pl_dist_sqr = torch.sum(wi * wi, dim=-1, keepdim=True)  # n_pts, 1
    wi_normalized = F.normalize(wi, dim=-1)

    # only refine camera poses
    wo = viewpoint_camera.camera_center - refine_means3D.detach()
    wo_normalized = F.normalize(wo, dim=-1)

    # use gaussian normal
    normal = pc.get_normal
    normal_wo_grad = normal.detach()

    # try highlight cues
    # ---------------------------------------------------------------

    # compute actual specular with pc's roughness
    roughness = pc.get_roughness

    hvec = F.normalize(wi_normalized + wo_normalized)  # n x 3
    n_dot_l = dot_prod(normal_wo_grad, wi_normalized).clip(1e-3, 1 - 1e-3)  # n x 1
    n_dot_v = dot_prod(normal_wo_grad, wo_normalized).clip(1e-3, 1 - 1e-3)
    n_dot_h = dot_prod(normal_wo_grad, hvec).clip(1e-3, 1 - 1e-3)
    h_dot_v = dot_prod(hvec, wo_normalized).clip(1e-3, 1 - 1e-3)
    n_dot_h_2 = n_dot_h ** 2

    # G
    k = (roughness + 1.) * (roughness + 1.) / 8.
    g1 = n_dot_v / (n_dot_v * (1. - k) + k)
    g2 = n_dot_l / (n_dot_l * (1. - k) + k)
    g = g1 * g2
    # N
    a2 = roughness * roughness
    ndf = a2 / (torch.pi * torch.pow((n_dot_h_2 * (a2 - 1.) + 1.), 2))
    # F
    f = 0.04 + 0.96 * torch.pow(1. - h_dot_v, 5)
    specular_cue = ndf * g * f / (4. * n_dot_v * n_dot_l + 1e-3)

    # ---------------------------------------------------------------

    # add ray tracing visibility condition (V1, shading gaussians)
    # ------------------------------------------------------------------------
    pts_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)[..., None].detach()
    pts_view = viewpoint_camera.pl_view_matrix @ pts_homo  # n x 4 x 1
    pts_depth = -pts_view[:, 2]  # n x 1
    pts_proj = viewpoint_camera.pl_proj_matrix @ pts_view
    pts_proj_xy = pts_proj[:, :2, 0] / pts_proj[:, 3:, 0]  # n x 2, range [-1, 1]

    query_depth = F.grid_sample(lgt_depth.reshape(1, 1, viewpoint_camera.image_height, viewpoint_camera.image_width),
                                pts_proj_xy.reshape(1, -1, 1, 2),
                                padding_mode='border',
                                align_corners=True).reshape(-1, 1)

    visibility = (query_depth >= (pts_depth * (1. - 1e-2))) * 1.

    # ------------------------------------------------------------------------

    # shading
    # ------------------------------------------------------------------------

    scatter = pc.scatter_net(wi_normalized, wo_normalized, pc.get_latcodes, normal_wo_grad, specular_cue)
    lgt_tspt = pc.incident_net(wi_normalized, pc.get_latcodes, visibility, n_dot_l)
    # intensity by square attenuation
    lgt_intensity = pc.get_pl_intensity / pl_dist_sqr
    color = scatter * lgt_tspt * lgt_intensity

    return color


def rasterize_color(viewpoint_camera: Camera, pc: GaussianModel, color, pipe,
           bg_color: torch.Tensor, scaling_modifier=1.0):
    """
    Raterize an image given Gaussians' color

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        depth_threshold=0.37,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    refine_means3D = means3D

    means2D = screenspace_points

    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(None, scaling_modifier)
    else:
        scales = pc.get_scaling
        if scales.shape[1] == 1:
            scales = scales.repeat(1, 3)

        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None

    # use gaussian normal
    normal = pc.get_normal

    raster_feats = torch.cat([color, normal * 0.5 + 0.5], dim=-1)  # n x 6

    rendered_feats, radii, pixels = rasterizer(
        means3D=refine_means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=raster_feats,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    rendered_image = linear2srgb(rendered_feats[:3])

    return rendered_image


def render(viewpoint_camera: Camera, pc : GaussianModel, lgt_depth, pose_refine, scene: Scene, pipe, bg_color : torch.Tensor,
           scaling_modifier = 1.0, override_color = None, get_components = False, save_color_to_sh = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        depth_threshold=0.37,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    pose_refine_mat = getWorld2ViewQuat(pose_refine[:4], pose_refine[4:]) if pose_refine is not None else None
    if pose_refine_mat is not None:
        means3D_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)[..., None] # n x 4 x 1
        refine_means3D = (pose_refine_mat @ means3D_homo)[:, :3, 0]
    else:
        refine_means3D = means3D

    # calculate wi&wo
    wi = viewpoint_camera.pl_pos - means3D.detach()
    pl_dist_sqr = torch.sum(wi * wi, dim=-1, keepdim=True)  # n_pts, 1
    pl_dist = torch.norm(wi, dim=-1, keepdim=True)
    wi_normalized = F.normalize(wi, dim=-1)

    # only refine camera poses
    wo = viewpoint_camera.camera_center - refine_means3D.detach()
    wo_normalized = F.normalize(wo, dim=-1)

    means2D = screenspace_points

    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(pose_refine[:3, :3], scaling_modifier)
    else:
        scales = pc.get_scaling
        if scales.shape[1] == 1:
            scales = scales.repeat(1, 3)

        rotations = pc.get_rotation

        if pose_refine is not None:
            quat_refine = F.normalize(pose_refine[:4].reshape(1, 4), dim=-1)
            rotations = quat_mul(quat_refine, rotations)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    components = {}

    # use gaussian normal
    normal = pc.get_normal
    normal_wo_grad = normal.detach()

    # try highlight cues from NRHints, currently it does not bring noticeable improvement
    # ---------------------------------------------------------------

    # compute actual specular with pc's roughness
    roughness = pc.get_roughness

    hvec = F.normalize(wi_normalized + wo_normalized) # n x 3
    n_dot_l = dot_prod(normal_wo_grad, wi_normalized).clip(1e-3, 1 - 1e-3) # n x 1
    n_dot_v = dot_prod(normal_wo_grad, wo_normalized).clip(1e-3, 1- 1e-3)
    n_dot_h = dot_prod(normal_wo_grad, hvec).clip(1e-3, 1 - 1e-3)
    h_dot_v = dot_prod(hvec, wo_normalized).clip(1e-3, 1 - 1e-3)
    n_dot_h_2 = n_dot_h ** 2

    # G
    k = (roughness + 1.) * (roughness + 1.) / 8.
    g1 = n_dot_v / (n_dot_v * (1. - k) + k)
    g2 = n_dot_l / (n_dot_l * (1. - k) + k)
    g = g1 * g2
    # N
    a2 = roughness * roughness
    ndf = a2 / (torch.pi * torch.pow((n_dot_h_2 * (a2 - 1.) + 1.), 2))
    # F
    f = 0.04 + 0.96 * torch.pow(1. - h_dot_v, 5)
    specular_cue = ndf * g * f / (4. * n_dot_v * n_dot_l + 1e-3)

    # ---------------------------------------------------------------

    # add ray tracing visibility condition (V1, shading gaussians)
    # ------------------------------------------------------------------------
    pts_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)[..., None].detach()
    pts_view = viewpoint_camera.pl_view_matrix @ pts_homo # n x 4 x 1
    pts_depth = -pts_view[:, 2] # n x 1
    pts_proj = viewpoint_camera.pl_proj_matrix @ pts_view
    pts_proj_xy = pts_proj[:, :2, 0] / pts_proj[:, 3:, 0] # n x 2, range [-1, 1]

    query_depth = F.grid_sample(lgt_depth.reshape(1, 1, viewpoint_camera.image_height, viewpoint_camera.image_width),
                                pts_proj_xy.reshape(1, -1, 1, 2),
                                padding_mode='border',
                                align_corners=True).reshape(-1, 1)

    visibility = (query_depth >= (pts_depth * (1.- 1e-2))) * 1.

    # ------------------------------------------------------------------------

    # shading
    # ------------------------------------------------------------------------

    rendered_comps = None

    # our shading
    scatter = pc.scatter_net(wi_normalized, wo_normalized, pc.get_latcodes, normal_wo_grad, specular_cue)
    incident = pc.incident_net(wi_normalized, pc.get_latcodes, visibility, n_dot_l)
    # intensity by square attenuation
    lgt_intensity = pc.get_pl_intensity / pl_dist_sqr
    color = scatter * incident * lgt_intensity

    # ------------------------------------------------------------------------

    if get_components:
        raster_comps = torch.cat([scatter, incident.repeat(1, 3)], dim=-1)
        rendered_comps, _, _ = rasterizer(
            means3D=refine_means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=raster_comps,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

    if override_color is not None:
        color = override_color

    raster_feats = torch.cat([color, normal * 0.5 + 0.5], dim=-1)  # n x 6

    rendered_feats, radii, pixels = rasterizer(
        means3D=refine_means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=raster_feats,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    rendered_image = linear2srgb(rendered_feats[:3])

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "normal": rendered_feats[3:6],
            "scatter": linear2srgb(rendered_comps[:3]) if rendered_comps is not None else None,
            "tspt": linear2srgb(rendered_comps[3:6]) if rendered_comps is not None else None,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "pixels": pixels}


