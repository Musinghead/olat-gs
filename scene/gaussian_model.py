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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, build_scaling_rotation2

from scene.networks import ScatterNet, IncidentNet


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, rot_refine):
            L = build_scaling_rotation2(scaling_modifier * scaling, rotation, rot_refine)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        # densify
        self.xyz_grad_peak = torch.empty(0)

        # olat
        self.lat_dim = 64
        self._latcodes = torch.empty(0)

        # trainable roughness, range [0, 1] to produce specular cue to scatter_net
        # currently, albedo and metalness are not used
        self._albedo = torch.empty(0)
        self._roughness = torch.empty(0)
        self._metalness = torch.empty(0)

        self._normal = torch.empty(0)

        # scatter mlp, (lc, wi, wo) -> scatter (3), range [0, +inf]
        self.scatter_net = ScatterNet(
            d_in=3,
            d_latcode=self.lat_dim,
            d_h=self.lat_dim * 2,
            d_out=3,
            encoding_cfg={
                "otype": "SphericalHarmonics",
                "degree": 4
            },
        )

        # (lc, wi) -> light incident factor (1), range [0, 1]
        self.incident_net = IncidentNet(
            d_in=3,
            d_latcode=self.lat_dim,
            d_h=self.lat_dim * 2,
            d_out=1,
            encoding_cfg={
                "otype": "SphericalHarmonics",
                "degree": 4
            },
        )

        # light intensity
        self._pl_intensity = torch.empty(0)

        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def set_all_requires_grad(self, requires=False):
        self._xyz.requires_grad_(requires)
        self._scaling.requires_grad_(requires)
        self._rotation.requires_grad_(requires)
        self._opacity.requires_grad_(requires)
        self._latcodes.requires_grad_(requires)
        self._normal.requires_grad_(requires)

        self._albedo.requires_grad_(requires)
        self._roughness.requires_grad_(requires)
        self._metalness.requires_grad_(requires)

        for param in self.scatter_net.parameters():
            param.requires_grad_(requires)

        for param in self.incident_net.parameters():
            param.requires_grad_(requires)


    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_normal(self):
        return F.normalize(self._normal, dim=-1)

    @property
    def get_roughness(self):
        return torch.sigmoid(self._roughness)

    @property
    def get_albedo(self):
        return torch.sigmoid(self._albedo)

    @property
    def get_metalness(self):
        return torch.sigmoid(self._metalness)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_latcodes(self):
        return self._latcodes

    @property
    def get_pl_intensity(self):
        return self._pl_intensity

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, rot_refine, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation, rot_refine)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, samples, spatial_lr_scale : float, pl_intensity: float):
        self.spatial_lr_scale = spatial_lr_scale
        # fused_point_cloud = torch.tensor(np.asarray(samples.points)).float().cuda()
        fused_point_cloud = samples['points'].clone()
        # fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        fused_color = torch.zeros_like(fused_point_cloud)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(samples.points)).float().cuda()), 0.0000001)
        dist2 = torch.clamp_min(distCUDA2(samples['points']), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # initialize latcode
        latcodes = torch.randn(fused_color.shape[0], self.lat_dim, device="cuda") * 0.001
        self._latcodes = nn.Parameter(latcodes.requires_grad_(True))

        # initialize from the mesh samples
        normal = samples['normals'].clone()
        self._normal = nn.Parameter(normal.requires_grad_(True))

        # initial roughness 0.15
        roughness = inverse_sigmoid(0.15 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self._roughness = nn.Parameter(roughness.requires_grad_(True))

        metalness = inverse_sigmoid(0.15 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self._metalness = nn.Parameter(metalness.requires_grad_(True))

        albedo = torch.zeros((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")
        self._albedo = nn.Parameter(albedo.requires_grad_(True))

        self._pl_intensity = pl_intensity
        print('call create_from_pcd, pl intensity: {}'.format(self.get_pl_intensity))

        print('init mean scale: {}'.format(self.get_scaling.detach().max(dim=-1).values.mean()))


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.xyz_grad_peak = torch.zeros((self.get_xyz.shape[0], 1), dtype=torch.bool, device="cuda")
        self.peak_start_iter = training_args.peak_start_iter

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            # OLAT
            {'params': [self._latcodes], 'lr': training_args.latcode_lr, "name": "latcode"},
            {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
            # GGX
            {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
            {'params': [self._albedo], 'lr': training_args.roughness_lr, "name": "albedo"},
            {'params': [self._metalness], 'lr': training_args.roughness_lr, "name": "metalness"},

            {'params': self.scatter_net.parameters(), 'lr': training_args.scatter_net_lr, "name": "net_scatter"},
            {'params': self.incident_net.parameters(), 'lr': training_args.incident_net_lr, "name": "net_incident"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.scatter_net_sched = get_expon_lr_func(lr_init=training_args.scatter_net_lr,
                                                    lr_final=training_args.scatter_net_lr * 0.1,
                                                    max_steps=training_args.iterations)
        self.incident_net_sched = get_expon_lr_func(lr_init=training_args.incident_net_lr,
                                                lr_final=training_args.incident_net_lr * 0.1,
                                                max_steps=training_args.iterations)
        self.latcode_sched = get_expon_lr_func(lr_init=training_args.latcode_lr,
                                                lr_final=training_args.latcode_lr * 0.1,
                                                max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "net_scatter":
                lr = self.scatter_net_sched(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "net_incident":
                lr = self.incident_net_sched(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "latcode":
                lr = self.latcode_sched(iteration)
                param_group['lr'] = lr
                # return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        for i in range(self._latcodes.shape[1]):
            l.append('latcode_{}'.format(i))

        l.append('roughness')
        for i in range(self._albedo.shape[1]):
            l.append('albedo_{}'.format(i))
        l.append('metalness')

        return l

    def construct_vanilla_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = self.get_normal.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        latcodes = self._latcodes.detach().cpu().numpy()

        roughness = self._roughness.detach().cpu().numpy()
        albedo = self._albedo.detach().cpu().numpy()
        metalness = self._metalness.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, latcodes, roughness, albedo, metalness), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        print('save trained mean scale: {}'.format(self.get_scaling.detach().max(dim=-1).values.mean()))
        print('save trained num pts: {}'.format(self._xyz.detach().shape[0]))


    def save_vanilla_ply(self, path):
        # write a .ply with vanilla gs params, to feed gaussian viewer
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_vanilla_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
                        np.asarray(plydata.elements[0]["ny"]),
                        np.asarray(plydata.elements[0]["nz"])),  axis=1)

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # latcodes
        lc_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("latcode")]
        lc_names = sorted(lc_names, key=lambda x: int(x.split('_')[-1]))
        latcodes = np.zeros((xyz.shape[0], len(lc_names)))
        for idx, attr_name in enumerate(lc_names):
            latcodes[:, idx] = np.asarray(plydata.elements[0][attr_name])

        roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]

        ab_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("albedo")]
        ab_names = sorted(ab_names, key=lambda x: int(x.split('_')[-1]))
        albedo = np.zeros((xyz.shape[0], len(ab_names)))
        for idx, attr_name in enumerate(ab_names):
            albedo[:, idx] = np.asarray(plydata.elements[0][attr_name])

        metalness = np.asarray(plydata.elements[0]["metalness"])[..., np.newaxis]

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self._latcodes = nn.Parameter(torch.tensor(latcodes, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))

        self._roughness = nn.Parameter(torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))
        self._albedo = nn.Parameter(torch.tensor(albedo, dtype=torch.float, device="cuda").requires_grad_(True))
        self._metalness = nn.Parameter(torch.tensor(metalness, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def load_nets(self, path):
        nets = torch.load(path)

        self.scatter_net.load_state_dict(nets["scatter"])
        self.incident_net.load_state_dict(nets["incident"])

        self._pl_intensity = nets["pl_intensity"]

    def save_nets(self, path):
        nets = {
            "scatter": self.scatter_net.state_dict(),
            "incident": self.incident_net.state_dict(),
            "pl_intensity": self.get_pl_intensity
        }
        torch.save(nets, path)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"].startswith("net"):
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._latcodes = optimizable_tensors["latcode"]
        self._normal = optimizable_tensors["normal"]

        self._roughness = optimizable_tensors["roughness"]
        self._albedo = optimizable_tensors["albedo"]
        self._metalness = optimizable_tensors["metalness"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self.xyz_grad_peak = self.xyz_grad_peak[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # only extend per-gaussian parameters, skip global networks
            if group["name"].startswith("net"):
                continue

            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest,
                              new_opacities, new_scaling, new_rotation,
                              new_latcodes, new_normal, new_roughness,
                              new_albedo, new_metalness):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "latcode": new_latcodes,
        "normal": new_normal,
        "roughness": new_roughness,
        "albedo": new_albedo,
        "metalness": new_metalness,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._latcodes = optimizable_tensors["latcode"]
        self._normal = optimizable_tensors["normal"]

        self._roughness = optimizable_tensors["roughness"]
        self._albedo = optimizable_tensors["albedo"]
        self._metalness = optimizable_tensors["metalness"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.xyz_grad_peak = torch.zeros((self.get_xyz.shape[0], 1), dtype=torch.bool, device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, grad_peak, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()

        padded_grad_peak = torch.zeros((n_init_points), dtype=torch.bool, device="cuda")
        padded_grad_peak[:grad_peak.shape[0]] = grad_peak.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, padded_grad_peak)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)


        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        new_latcodes = self._latcodes[selected_pts_mask].repeat(N,1)
        new_normal = self._normal[selected_pts_mask].repeat(N,1)

        new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)
        new_albedo = self._albedo[selected_pts_mask].repeat(N, 1)
        new_metalness = self._metalness[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                   new_opacity, new_scaling, new_rotation,
                                   new_latcodes, new_normal, new_roughness,
                                   new_albedo, new_metalness)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, grad_peak):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, grad_peak)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_latcodes = self._latcodes[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]

        new_roughness = self._roughness[selected_pts_mask]
        new_albedo = self._albedo[selected_pts_mask]
        new_metalness = self._metalness[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                   new_opacities, new_scaling, new_rotation,
                                   new_latcodes, new_normal, new_roughness,
                                   new_albedo, new_metalness)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iter):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # densify a gaussian if its grad is over peak in any view
        if iter >= self.peak_start_iter:
            grad_peak = self.xyz_grad_peak.clone().squeeze()
        else:
            grad_peak = torch.zeros_like(self.xyz_grad_peak).squeeze()

        self.densify_and_clone(grads, max_grad, extent, grad_peak)
        self.densify_and_split(grads, max_grad, extent, grad_peak)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, pixels, iter):
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)

        self.xyz_gradient_accum[update_filter] += grad_norm * pixels[update_filter]
        self.denom[update_filter] += pixels[update_filter]

        if iter >= self.peak_start_iter:
             # densify a gaussian if its grad is over a peak value in any view
             # to (slightly) improve shadow edge quality
            is_over_peak = grad_norm >= 0.0008
            self.xyz_grad_peak[update_filter] = torch.logical_or(self.xyz_grad_peak[update_filter], is_over_peak)