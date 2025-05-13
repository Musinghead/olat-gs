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
import numpy as np
from typing import NamedTuple
from utils.general_utils import build_rotation

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getWorld2ViewQuat(R, t):
    """
    R: quaternion
    """
    Rt = torch.eye(4, dtype=torch.float32, device='cuda')
    rot = build_rotation(R.reshape(1,4)).squeeze()
    Rt[:3, :3] = rot
    Rt[:3, 3] = t
    return Rt

def getWorld2ViewGL(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    """
    OpenGL format, +y -> up, +z -> back
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    # cv to gl axis, flip y&z
    C2W[:3, 1:3] *= -1
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getLgtView(pl_pos):
    '''
    generate world to view transform matrix at the point light view
    following OpenGL format, +x -> right, +y -> up, +z -> back
    '''
    Rt = torch.eye(4, dtype=torch.float32, device='cuda')

    center = pl_pos.reshape(1, 3)
    up = torch.tensor([[0., 1., 0.]], dtype=torch.float32, device='cuda')
    back = F.normalize(center, dim=-1)
    right = F.normalize(torch.cross(up, back, dim=-1), dim=-1)
    up_revise = F.normalize(torch.cross(back, right, dim=-1), dim=-1)

    Rt[:3] = torch.cat([right, up_revise, back, center], dim=0).T
    Rt_w2c = torch.linalg.inv(Rt)
    return Rt_w2c


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    # different from std opengl, this setting will transform z from [n, f] to [0, 1]
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrix2(znear, zfar, fovX, fovY, cx, cy, width, height):
    fx = fov2focal(fovX, width)
    fy = fov2focal(fovY, height)

    P = torch.zeros(4, 4)

    z_sign = 1.0
    P[0, 0] = 2. * fx / width
    P[1, 1] = 2. * fy / height

    P[0, 2] = -(width - 2. * cx) / width
    P[1, 2] = -(height - 2. * cy) / height
    P[3, 2] = z_sign
    # different from std opengl, this setting will transform z from [n, f] to [0, 1]
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getOpenGLProjMatrix(znear, zfar, fx, fy, cx, cy, width, height):
    """
    generate an OpenGL format perspective projection matrix, map points from view space to clip space
    i.e., transform points in view frustum to [-1, 1]
    """
    P = torch.zeros(4, 4, dtype=torch.float32, device='cuda')

    z_sign = -1.0
    P[0, 0] = 2. * fx / width
    P[1, 1] = -2. * fy / height

    P[0, 2] = (width - 2. * cx) / width
    P[1, 2] = (height - 2. * cy) / height
    P[3, 2] = z_sign

    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = -(2. * zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))