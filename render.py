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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, scene: Scene):

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal")
    mesh_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mesh_normal")

    incident_path = os.path.join(model_path, name, "ours_{}".format(iteration), "incident")
    scatter_path = os.path.join(model_path, name, "ours_{}".format(iteration), "scatter")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    makedirs(mesh_normal_path, exist_ok=True)

    makedirs(incident_path, exist_ok=True)
    makedirs(scatter_path, exist_ok=True)

    lambda_dssim = 0.2
    first_iter = 0
    last_iter = 200
    psnr_refine_total = 0.

    if not pipeline.no_refine_pose:
        print("> enable test camera pose registration")

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # register test pose
        if not pipeline.no_refine_pose and name.startswith('test'):
            gt_image = view.original_image[0:3, :, :].cuda()
            for iter in range(first_iter, last_iter):
                pose_refine = scene.getTestRefinePose(view.colmap_id)
                lgt_depth = scene.render_lgt_depth(view)
                render_pkg = render(view, gaussians, lgt_depth, pose_refine, scene, pipeline, background, get_components=False)
                image = render_pkg["render"]

                Ll1 = l1_loss(image, gt_image)
                loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image))
                loss.backward()

                scene.optimizer.step()
                scene.optimizer.zero_grad(set_to_none=True)

        save_gt_image = view.original_image[0:3, :, :]

        save_pose_refine = scene.getTestRefinePose(view.colmap_id) if name.startswith('test') else scene.getTrainRefinePose(view.colmap_id)
        if pipeline.no_refine_pose:
            save_pose_refine = None
        save_lgt_depth = scene.render_lgt_depth(view)
        save_render_pkg = render(view, gaussians, save_lgt_depth, save_pose_refine, scene, pipeline, background, get_components=True)

        psnr_refine = psnr(save_render_pkg["render"].clip(0, 1), save_gt_image.clip(0, 1).cuda()).mean().cpu().item()
        psnr_refine_total += psnr_refine

        save_gt_normal = scene.render_normal(view, save_pose_refine)

        torchvision.utils.save_image(save_render_pkg["render"], os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(save_gt_image, os.path.join(gts_path, view.image_name + ".png"))
        torchvision.utils.save_image(save_render_pkg["normal"], os.path.join(normal_path, view.image_name + ".png"))
        torchvision.utils.save_image(save_gt_normal, os.path.join(mesh_normal_path, view.image_name + ".png"))

        if save_render_pkg["scatter"] is not None:
            torchvision.utils.save_image(save_render_pkg["scatter"], os.path.join(scatter_path, view.image_name + ".png"))
        if save_render_pkg["tspt"] is not None:
            torchvision.utils.save_image(save_render_pkg["tspt"], os.path.join(incident_path, view.image_name + ".png"))

    print('avg register psnr {}'.format(psnr_refine_total / len(views)))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, scene)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, scene)


def render_sets2(dataset : ModelParams, iteration : int, pipeline : PipelineParams, op: OptimizationParams, skip_train : bool, skip_test : bool, postfix: str):

    gaussians = GaussianModel(dataset.sh_degree)
    # freeze all gaussian parameters
    gaussians.set_all_requires_grad(False)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_train_img=skip_train)

    print('num pts {}, mean max scale {}'.format(gaussians.get_xyz.shape[0], gaussians.get_scaling.max(dim=1).values.mean()))
    print('loaded {} train views, {} test views'.format(len(scene.getTrainCameras()), len(scene.getTestCameras())))

    # refine test camera pose
    scene.training_setup(op)

    # freeze all params except test_pose_refine
    scene.set_all_requires_grad(False)
    scene.setTestRefinePoseGrad(True)

    n_chan = 6
    bg_color = [0 for _ in range(n_chan)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    if dataset.white_background:
        background[:3] = 1.

    if not skip_train:
        render_set(dataset.model_path, "train"+postfix, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, scene)

    if not skip_test:
        render_set(dataset.model_path, "test"+postfix, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, scene)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--postfix", "-p", default='', type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets2(model.extract(args), args.iteration, pipeline.extract(args), op.extract(args), args.skip_train, args.skip_test, args.postfix)