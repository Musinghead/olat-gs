"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Rays
from tqdm import tqdm
import math

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def _load_renderings(root_fp: str, subject_id: str, split: str, cam_scale: float=1.0, num_lmt: int=1000, interval: int=1):
    """Load images from disk."""

    data_dir = os.path.join(root_fp, subject_id)
    with open(
        os.path.join(data_dir, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []
    pl_pos = []

    # load up to 1,000 views
    for i in tqdm(range(0, min(len(meta["frames"]), num_lmt), interval)):
        frame = meta["frames"][i]
        
        fname = os.path.join(data_dir, frame["file_path"] + ".png")
        rgba = imageio.imread(fname)
        camtoworlds.append(frame["transform_matrix"])
        images.append(rgba)
        pl_pos.append(frame['pl_pos'])

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)
    # gl to cv
    camtoworlds[:, :3, 1:3] *= -1
    pl_pos = np.stack(pl_pos, axis=0)

    h, w = images.shape[1:3]

    if "camera_intrinsics" in meta.keys():
        print("> read calibrated intrinsics")
        cx, cy, fx, fy = meta["camera_intrinsics"]
        K = torch.tensor([
            [fx, 0., cx],
            [0., fy, cy],
            [0., 0., 1.]
        ], dtype=torch.float32)
    else:
        # synthetic
        print("> read synthetic intrinsics")
        camera_angle_x = meta["camera_angle_x"]
        fx = fov2focal(camera_angle_x, w)
        fy = fx
        cx = w / 2
        cy = h / 2
        K = torch.tensor([
            [fx, 0., cx],
            [0., fy, cy],
            [0., 0., 1.]
        ], dtype=torch.float32)

    # scale scene
    camtoworlds[:, :3, 3] *= cam_scale
    pl_pos *= cam_scale

    return images, camtoworlds, K, pl_pos

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

# This implementation is borrowed from NRHints: https://github.com/iamNCJ/NRHints
def exp_map_SO3xR3(tangent_vector):
    """Compute the exponential map of the direct product group `SO(3) x R^3`.

    This can be used for learning pose deltas on SE(3), and is generally faster than `exp_map_SE3`.

    Args:
        tangent_vector: Tangent vector; length-3 translations, followed by an `so(3)` tangent vector.
    Returns:
        [R|t] transformation matrices.
    """
    # code for SO3 map grabbed from pytorch3d and stripped down to bare-bones
    log_rot = tangent_vector[:, 3:]
    nrms = (log_rot * log_rot).sum(1)
    rot_angles = torch.clamp(nrms, 1e-4).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = torch.zeros((log_rot.shape[0], 3, 3), dtype=log_rot.dtype, device=log_rot.device)
    skews[:, 0, 1] = -log_rot[:, 2]
    skews[:, 0, 2] = log_rot[:, 1]
    skews[:, 1, 0] = log_rot[:, 2]
    skews[:, 1, 2] = -log_rot[:, 0]
    skews[:, 2, 0] = -log_rot[:, 1]
    skews[:, 2, 1] = log_rot[:, 0]
    skews_square = torch.bmm(skews, skews)

    ret = torch.zeros(tangent_vector.shape[0], 3, 4, dtype=tangent_vector.dtype, device=tangent_vector.device)
    ret[:, :3, :3] = (
            fac1[:, None, None] * skews
            + fac2[:, None, None] * skews_square
            + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    # Compute the translation
    ret[:, :3, 3] = tangent_vector[:, :3]
    return ret


class CamPoseResi(nn.Module):
    def __init__(self, n_views):
        super().__init__()
        self.n_views = n_views
        # cam pose residual
        self.cam_params = nn.Parameter(torch.zeros((self.n_views, 6), device='cuda', dtype=torch.float32).requires_grad_(True))

    def apply_resi(self, image_id, c2w_mat):
        cam_resi = exp_map_SO3xR3(self.cam_params[image_id])
        dR = cam_resi[:, :3, :3]
        dt = cam_resi[:, :3, 3:]

        R = dR.matmul(c2w_mat[:, :3, :3])
        t = dt + dR.matmul(c2w_mat[:, :3, 3:])

        return R, t.squeeze(-1)


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test"]

    OPENGL_CAMERA = False

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "black",
        num_rays: int = None,
        batch_over_images: bool = True,
        device: torch.device = torch.device("cpu"),
        camopt: bool = False,
        cam_scale: float = 1.0
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.num_rays = num_rays

        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        if split == "trainval":
            _images_train, _camtoworlds_train, _focal_train, _pl_pos_train = _load_renderings(
                root_fp, subject_id, "train", cam_scale
            )
            _images_val, _camtoworlds_val, _focal_val, _pl_pos_val = _load_renderings(
                root_fp, subject_id, "val", cam_scale
            )
            self.images = np.concatenate([_images_train, _images_val])
            self.camtoworlds = np.concatenate(
                [_camtoworlds_train, _camtoworlds_val]
            )
            self.focal = _focal_train
            self.pl_pos = _pl_pos_train
        elif split == "train":
            self.images, self.camtoworlds, self.focal, self.pl_pos = _load_renderings(
                root_fp, subject_id, split, cam_scale
            )
        else:
            # test split, only load 2 views to test visualization
            self.images, self.camtoworlds, self.focal, self.pl_pos = _load_renderings(
                root_fp, subject_id, split, cam_scale, 61, 60
            )
        self.images = torch.from_numpy(self.images).to(torch.uint8)
        self.HEIGHT, self.WIDTH = self.images.shape[1:3]

        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.K = self.focal

        self.images = self.images.to(device)
        self.camtoworlds = self.camtoworlds.to(device)
        self.K = self.K.to(device)

        self.pl_pos = torch.from_numpy(self.pl_pos).to(torch.float32).to(device)

        self.cam_pose_resi = CamPoseResi(len(self.images))
        self.camopt = camopt

        assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)
        self.g = torch.Generator(device=device)
        self.g.manual_seed(42)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(
                    3, device=self.images.device, generator=self.g
                )
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)
        else:
            # just use white during inference
            # color_bkgd = torch.ones(3, device=self.images.device)
            color_bkgd = torch.zeros(3, device=self.images.device)

        # color_bkgd currently is not used
        # pixels = pixels * alpha + color_bkgd * (1.0 - alpha)

        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "alpha": alpha, # [n_rays, 1]
            "rays": rays,  # [n_rays,] or [h, w]
            # "color_bkgd": color_bkgd,  # [3,]
            "color_bkgd": None,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index, apply_camopt=False):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(num_rays,),
                    device=self.images.device,
                    generator=self.g,
                )
            else:
                image_id = [index] * num_rays
            x = torch.randint(
                0,
                self.WIDTH,
                size=(num_rays,),
                device=self.images.device,
                generator=self.g,
            )
            y = torch.randint(
                0,
                self.HEIGHT,
                size=(num_rays,),
                device=self.images.device,
                generator=self.g,
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.images.device),
                torch.arange(self.HEIGHT, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        rgba = self.images[image_id, y, x] / 255.0  # (num_rays, 4)
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)

        # apply cam pose residual
        if self.camopt and apply_camopt:
            R, t = self.cam_pose_resi.apply_resi(image_id, c2w)
        else:
            R = c2w[:, :3, :3]
            t = c2w[:, :3, -1]

        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]
        pl_pos = self.pl_pos[image_id] # (num_rays, 3) or (1, 3)

        directions = (camera_dirs[:, None, :] * R).sum(dim=-1)
        origins = torch.broadcast_to(t, directions.shape)
        
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))

            rgba = torch.reshape(rgba, (num_rays, 4))
            pl_pos = torch.reshape(pl_pos, (num_rays, 3))
        else:
            # no need for optimize test view poses
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))

            rgba = torch.reshape(rgba, (self.HEIGHT, self.WIDTH, 4))
            pl_pos = torch.reshape(pl_pos, (1, 1, 3)).expand(self.HEIGHT, self.WIDTH, -1)
        
        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "pl_pos": pl_pos, # [h, w, 3] or [num_rays, 3]
        }
