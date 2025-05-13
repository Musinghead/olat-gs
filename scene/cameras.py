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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix2, getLgtView, getOpenGLProjMatrix, fov2focal, getWorld2ViewGL
from utils.general_utils import build_rotation


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, cx, cy, pl_pos, image, width, height, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        # this pose follows opencv format, +y down, +z forward
        self.R = R
        self.T = T

        self.FoVx = FoVx
        self.FoVy = FoVy
        self.cx = cx
        self.cy = cy

        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.original_image = image.clamp(0.0, 1.0) if image is not None else None
        self.image_width = width
        self.image_height = height

        self.fx = fov2focal(self.FoVx, self.image_width)
        self.fy = fov2focal(self.FoVy, self.image_height)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # for gs rasterizer, OpenCV format, w2c, transposed
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        # include cx&cy to be more general, also handle cameras whose optical centers do not match image centers
        self.projection_matrix = getProjectionMatrix2(znear=self.znear, zfar=self.zfar,
                                                     fovX=self.FoVx, fovY=self.FoVy,
                                                     cx=self.cx, cy=self.cy,
                                                     width=self.image_width, height=self.image_height).transpose(0, 1).cuda()

        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # camera transformation in OpenGL formats
        self.cam_gl_view_mat = torch.tensor(getWorld2ViewGL(R, T, trans, scale)).cuda()
        self.cam_gl_proj_mat = getOpenGLProjMatrix(znear=self.znear, zfar=self.zfar,
                                                   fx=self.fx, fy=self.fy,
                                                   cx=self.cx, cy=self.cy,
                                                   width=self.image_width, height=self.image_height)
        self.cam_gl_fullproj = self.cam_gl_proj_mat @ self.cam_gl_view_mat

        # camera trans in OpenCV format
        cam_cv_intrin = torch.tensor([
            [self.fx, 0, cx],
            [0, self.fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device='cuda')
        self.cam_cv_intrin_inv = torch.linalg.inv(cam_cv_intrin)

        self.cam_cv_pose_c2w = torch.linalg.inv(self.world_view_transform.transpose(0, 1))

        # light views
        self.pl_pos = torch.tensor((pl_pos + trans) * scale, dtype=torch.float32).cuda()
        self.pl_view_matrix = getLgtView(self.pl_pos)
        # adjust focal length used at the light view, w.r.t the distance of light over the distance of camera
        # set the focal length slightly wider to include the whole subject
        focal_ratio = torch.norm(self.pl_pos) / torch.norm(self.camera_center) * 0.7
        self.pl_fx = focal_ratio * self.fx
        self.pl_fy = focal_ratio * self.fy
        self.pl_proj_matrix = getOpenGLProjMatrix(znear=self.znear, zfar=self.zfar,
                                                 fx=self.pl_fx, fy=self.pl_fy,
                                                 cx=self.cx, cy=self.cy,
                                                 width=self.image_width, height=self.image_height)
        self.pl_fullproj = self.pl_proj_matrix @ self.pl_view_matrix


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

