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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel, BasicPointCloud
from scene.cameras import Camera
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import nvdiffrast.torch as dr
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.graphics_utils import getWorld2ViewQuat
import shutil



class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, fix_view=-1, skip_train_img=False, skip_test_img=False, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            # read Olat data by default
            # if need to run on data of constant lighting, run the original GS project
            print("Found transforms_train.json file, assuming Olat data set!")
            scene_info = sceneLoadTypeCallbacks["Olat"](args.source_path, args.white_background, args.eval, args.num_view_limit, -1, fix_view, skip_train_img, skip_test_img)
        else:
            assert False, "Could not recognize scene type!"

        # nvdiffrast
        # gl context works on Windows well but not on Linux, due to compatibility issues
        # self.glctx = dr.RasterizeGLContext()
        self.glctx = dr.RasterizeCudaContext()
        self.proxy_mesh = scene_info.proxy_mesh

        # write to output folder
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # self.cameras_extent = scene_info.nerf_normalization["radius"]
        # currently olat scenes are normalized to [-1, 1] space
        self.cameras_extent = 1.
        print('[camera extent] {}'.format(self.cameras_extent))
        self.pl_intensity = scene_info.nerf_normalization["pl_intensity"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.nerf_normalization)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.nerf_normalization)

        # pose correction
        n_train_cam = len(self.train_cameras[1.0])
        self.train_pose_refine = torch.zeros((n_train_cam, 7), dtype=torch.float32, device='cuda')
        self.train_pose_refine[:, 0] = 1.

        n_test_cam = len(self.test_cameras[1.0])
        self.test_pose_refine = torch.zeros((n_test_cam, 7), dtype=torch.float32, device='cuda')
        self.test_pose_refine[:, 0] = 1.

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud_olat.ply"))
            # load global networks
            self.gaussians.load_nets(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "nets.pth"))
            # use the same intensity as the trained model
            self.pl_intensity = self.gaussians.get_pl_intensity
            scene_params_path = os.path.join(self.model_path,
                                       "point_cloud",
                                       "iteration_" + str(self.loaded_iter),
                                       "scene.pth")
            if os.path.exists(scene_params_path):
                scene_params = torch.load(scene_params_path)
                self.train_pose_refine = scene_params['train_pose_refine']

        else:
            # directly sample points from mesh, save at pcd
            rand_indices = torch.randperm(len(self.proxy_mesh['vertices']))[:args.num_init_samples]

            samples = {
                'points': self.proxy_mesh['vertices'][rand_indices][:, :3, 0], # n x 3
                'normals': self.proxy_mesh['normals'][0][rand_indices] # n x 3
            }

            self.gaussians.create_from_pcd(samples, self.cameras_extent, self.pl_intensity)

    def training_setup(self, training_args):
        params = [
            {'params': [self.train_pose_refine], 'lr': training_args.pose_opt_lr},
            {'params': [self.test_pose_refine], 'lr': training_args.pose_opt_lr}
        ]
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)

    def set_all_requires_grad(self, requires=False):
        self.train_pose_refine.requires_grad_(requires)
        self.test_pose_refine.requires_grad_(requires)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud_olat.ply"))
        # save global networks
        self.gaussians.save_nets(os.path.join(point_cloud_path, "nets.pth"))

        # scene params
        scene_params = {
            'train_pose_refine': self.train_pose_refine,
        }
        torch.save(scene_params, os.path.join(point_cloud_path, "scene.pth"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getTrainRefinePose(self, colmap_id):
        pose_refine = self.train_pose_refine[colmap_id]
        if colmap_id == 0:
            # keep train frame 0 static
            pose_refine = torch.zeros_like(pose_refine.detach())
            pose_refine[0] = 1.
        return pose_refine

    def getTestRefinePose(self, colmap_id):
        pose_refine = self.test_pose_refine[colmap_id]
        return pose_refine

    def setTrainRefinePoseGrad(self, requires=True):
        self.train_pose_refine.requires_grad_(requires)

    def setTestRefinePoseGrad(self, requires=True):
        self.test_pose_refine.requires_grad_(requires)

    def render_lgt_depth(self, view_cam):
        with torch.no_grad():
            pl_proj = view_cam.pl_proj_matrix
            pl_fullproj = view_cam.pl_fullproj

            mesh_proj_pts = pl_fullproj @ self.proxy_mesh['vertices']
            la = -pl_proj[2, 2]
            lb = -pl_proj[2, 3]

            lgt_rast, _ = dr.rasterize(self.glctx,
                                       mesh_proj_pts[None, :, :, 0],
                                       self.proxy_mesh['faces'],
                                       resolution=(view_cam.image_height, view_cam.image_width))

            z_w = lgt_rast[0, :, :, 2]
            tid = lgt_rast[0, :, :, 3]
            # set depth +inf at empty space
            lgt_depth = torch.where(tid != 0, -lb / (z_w - la), 1e4)
            return lgt_depth

    def render_normal(self, view_cam, pose_refine):
        # render normals in world coordinate
        with torch.no_grad():
            cam_proj = view_cam.cam_gl_proj_mat
            cam_la = cam_proj[2, 2]
            cam_lb = cam_proj[2, 3]

            cam_fullproj = view_cam.cam_gl_fullproj

            if pose_refine is not None:
                pose_refine_mat = getWorld2ViewQuat(pose_refine[:4], pose_refine[4:])
                mesh_proj_pts = cam_fullproj @ pose_refine_mat @ self.proxy_mesh['vertices']
            else:
                mesh_proj_pts = cam_fullproj @ self.proxy_mesh['vertices']
            cam_rast, _  = dr.rasterize(self.glctx,
                                        mesh_proj_pts[None , :, :, 0],
                                        self.proxy_mesh['faces'],
                                        resolution=(view_cam.image_height, view_cam.image_width))
            # [-1, 1] -> [0, 1]
            attr_map, _ = dr.interpolate(self.proxy_mesh['normals'] * 0.5 + 0.5, cam_rast, self.proxy_mesh['faces'])

            normal_map = attr_map[0, :, :, :3].clip(0, 1).permute(2, 0, 1)
            return normal_map



