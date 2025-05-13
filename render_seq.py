import torch
import torch.nn.functional as F
from scene import Scene
import os
from tqdm import tqdm
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
from scene.cameras import Camera
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from time import time


def spher2cart(radius, lati, longi):
    y = radius * torch.sin(lati / 180. * torch.pi) # up
    x = radius * torch.cos(lati / 180. * torch.pi) * torch.cos(longi / 180. * torch.pi)
    z = radius * torch.cos(lati / 180. * torch.pi) * torch.sin(longi / 180. * torch.pi)
    return torch.tensor([[x, y, z]], dtype=torch.float32, device='cuda')

def spher2cart_np(radius, lati, longi):
    y = radius * np.sin(lati / 180. * torch.pi) # up
    x = radius * np.cos(lati / 180. * torch.pi) * np.cos(longi / 180. * torch.pi)
    z = radius * np.cos(lati / 180. * torch.pi) * np.sin(longi / 180. * torch.pi)
    return np.array([[x, y, z]], dtype=np.float32)

def get_cam_view_mat(cam_pos, down):
    cam_rt = torch.eye(4, dtype=torch.float32, device='cuda')
    cam_lookat = -F.normalize(cam_pos.reshape(1, 3), dim=-1)
    cam_right = F.normalize(torch.cross(down, cam_lookat, dim=-1), dim=-1)
    cam_down = F.normalize(torch.cross(cam_lookat, cam_right, dim=-1), dim=-1)
    cam_rt[:3] = torch.cat([cam_right, cam_down, cam_lookat, cam_pos], dim=0).T

    return cam_rt


def render_sequence(dataset : ModelParams, iteration : int, pipeline : PipelineParams, inter, begin_longi, components):
    """
    fix camera views, rotate point light; rotate camera views, fix point light
    Returns:

    """
    dataset.eval = True
    gaussians = GaussianModel(dataset.sh_degree)

    # add an option to not read images
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_train_img=True, skip_test_img=True)

    n_chan = 6
    bg_color = [1 for _ in range(n_chan)] if dataset.white_background else [0 for _ in range(n_chan)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # generate a sequence of camera views
    train_cameras = scene.getTrainCameras()

    ref_view_cam = train_cameras[0]

    down = torch.tensor([[0., -1., 0.]], dtype=torch.float32, device='cuda')
    # avg distance
    cam_radius = torch.stack([cam.camera_center for cam in train_cameras], dim=0).norm(dim=-1).mean() * 1.
    pl_radius = torch.stack([cam.pl_pos for cam in train_cameras], dim=0).norm(dim=-1).mean() * 1.
    # radius_ratio_range = 0.8

    print('cam radius', cam_radius)
    print('pl radius', pl_radius)

    if inter:
        def nothing(x):
            pass
        cv2.namedWindow('control')
        cv2.resizeWindow('control', 300, 512)
        cv2.createTrackbar('cam longi', 'control', 0, 360, nothing)
        cv2.createTrackbar('cam lati', 'control', 45, 85, nothing)
        cv2.createTrackbar('lgt longi', 'control', 0, 360, nothing)
        cv2.createTrackbar('lgt lati', 'control', 45, 85, nothing)
        cv2.namedWindow('OLAT Gaussians')
        while True:
            cam_longi = cv2.getTrackbarPos('cam longi', 'control')
            cam_lati = cv2.getTrackbarPos('cam lati', 'control')

            lgt_longi = cv2.getTrackbarPos('lgt longi', 'control')
            lgt_lati = cv2.getTrackbarPos('lgt lati', 'control')

            cam_longi = torch.tensor([float(cam_longi)], dtype=torch.float32, device='cuda')
            cam_lati = torch.tensor([float(cam_lati)], dtype=torch.float32, device='cuda')

            lgt_longi = torch.tensor([float(lgt_longi)], dtype=torch.float32, device='cuda')
            lgt_lati = torch.tensor([float(lgt_lati)], dtype=torch.float32, device='cuda')

            # camera position, light position
            cam_pos = spher2cart(cam_radius, cam_lati, cam_longi)  # 1 x 3
            cam_c2w = get_cam_view_mat(cam_pos, down)
            cam_pose_w2c = torch.linalg.inv(cam_c2w)
            R = cam_pose_w2c[:3, :3].T
            T = cam_pose_w2c[:3, 3]

            pl_pos = spher2cart(pl_radius, lgt_lati, lgt_longi).cpu().numpy()

            sample_cam = Camera(colmap_id=0, R=R.cpu().numpy(), T=T.cpu().numpy(), FoVx=ref_view_cam.FoVx,
                                FoVy=ref_view_cam.FoVy, cx=ref_view_cam.cx, cy=ref_view_cam.cy,
                                pl_pos=pl_pos, image=None, width=ref_view_cam.image_width, height=ref_view_cam.image_height,
                                gt_alpha_mask=None, image_name=None, uid=0)

            lgt_depth = scene.render_lgt_depth(sample_cam)
            image = render(sample_cam, gaussians, lgt_depth, None, scene, pipeline, background)['render']

            image_np = (image.permute(1, 2, 0) * 255).clip(0, 255).byte().cpu().numpy()
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            cv2.imshow('OLAT Gaussians', image_np)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyAllWindows()
    else:
        # save 2 sequences, 1 fix cam rot lgt, 1 rot cam fix lgt
        print('> render and save a video')
        # fixed latitude
        latitude = torch.tensor([45.], dtype=torch.float32, device='cuda')
        image_list = []
        tspt_list = []
        scatter_list = []

        # seq 1, fix camera, rotate light
        # fix_cam_longi = begin_longi
        for i in tqdm(range(0 + begin_longi, 360 + begin_longi, 2)):
            lgt_longi = torch.tensor([float(i)], dtype=torch.float32, device='cuda')
            cam_longi = torch.tensor([float(begin_longi)], dtype=torch.float32, device='cuda')

            # camera position, light position
            cam_pos = spher2cart(cam_radius, latitude, cam_longi)  # 1 x 3
            cam_c2w = get_cam_view_mat(cam_pos, down)
            cam_pose_w2c = torch.linalg.inv(cam_c2w)
            R = cam_pose_w2c[:3, :3].T
            T = cam_pose_w2c[:3, 3]

            pl_pos = spher2cart(pl_radius, latitude, lgt_longi).cpu().numpy()

            sample_cam = Camera(colmap_id=0, R=R.cpu().numpy(), T=T.cpu().numpy(), FoVx=ref_view_cam.FoVx,
                                FoVy=ref_view_cam.FoVy, cx=ref_view_cam.cx, cy=ref_view_cam.cy,
                                pl_pos=pl_pos, image=None, width=ref_view_cam.image_width,
                                height=ref_view_cam.image_height,
                                gt_alpha_mask=None, image_name=None, uid=0)

            lgt_depth = scene.render_lgt_depth(sample_cam)
            render_res = render(sample_cam, gaussians, lgt_depth, None, scene, pipeline, background, get_components=components)

            image_list.append(render_res['render'].clone())

            if components:
                tspt = render_res['tspt']
                tspt_np = (tspt.permute(1, 2, 0) * 255).clip(0, 255).byte().cpu().numpy()
                tspt_np = cv2.cvtColor(tspt_np, cv2.COLOR_BGR2RGB)
                tspt_list.append(tspt_np)

                scatter = render_res['scatter']
                scatter_np = (scatter.permute(1, 2, 0) * 255).clip(0, 255).byte().cpu().numpy()
                scatter_np = cv2.cvtColor(scatter_np, cv2.COLOR_BGR2RGB)
                scatter_list.append(scatter_np)

        # seq 2, fix light, rotate camera
        for i in tqdm(range(0 + begin_longi, 360 + begin_longi, 2)):
            lgt_longi = torch.tensor([float(begin_longi)], dtype=torch.float32, device='cuda')
            cam_longi = torch.tensor([float(i)], dtype=torch.float32, device='cuda')

            # camera position, light position
            cam_pos = spher2cart(cam_radius, latitude, cam_longi)  # 1 x 3
            cam_c2w = get_cam_view_mat(cam_pos, down)
            cam_pose_w2c = torch.linalg.inv(cam_c2w)
            R = cam_pose_w2c[:3, :3].T
            T = cam_pose_w2c[:3, 3]

            pl_pos = spher2cart(pl_radius, latitude, lgt_longi).cpu().numpy()

            sample_cam = Camera(colmap_id=0, R=R.cpu().numpy(), T=T.cpu().numpy(), FoVx=ref_view_cam.FoVx,
                                FoVy=ref_view_cam.FoVy, cx=ref_view_cam.cx, cy=ref_view_cam.cy,
                                pl_pos=pl_pos, image=None, width=ref_view_cam.image_width,
                                height=ref_view_cam.image_height,
                                gt_alpha_mask=None, image_name=None, uid=0)

            lgt_depth = scene.render_lgt_depth(sample_cam)
            render_res = render(sample_cam, gaussians, lgt_depth, None, scene, pipeline, background,
                                get_components=components)

            image_list.append(render_res['render'].clone())

            if components:
                tspt = render_res['tspt']
                tspt_np = (tspt.permute(1, 2, 0) * 255).clip(0, 255).byte().cpu().numpy()
                tspt_np = cv2.cvtColor(tspt_np, cv2.COLOR_BGR2RGB)
                tspt_list.append(tspt_np)

                scatter = render_res['scatter']
                scatter_np = (scatter.permute(1, 2, 0) * 255).clip(0, 255).byte().cpu().numpy()
                scatter_np = cv2.cvtColor(scatter_np, cv2.COLOR_BGR2RGB)
                scatter_list.append(scatter_np)

        vout = cv2.VideoWriter(os.path.join(dataset.model_path, 'seq.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 60, (ref_view_cam.image_width, ref_view_cam.image_height))
        for idx in tqdm(range(len(image_list))):
            image = image_list[idx]
            image_np = (image.permute(1, 2, 0) * 255).clip(0, 255).byte().cpu().numpy()
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            vout.write(image_np)

        vout.release()

        if components:
            vout_tspt = cv2.VideoWriter(os.path.join(dataset.model_path, 'seq_tspt.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 60,
                                   (ref_view_cam.image_width, ref_view_cam.image_height))
            for idx in tqdm(range(len(tspt_list))):
                tspt_np = tspt_list[idx]
                vout_tspt.write(tspt_np)

            vout_tspt.release()

            vout_scatter = cv2.VideoWriter(os.path.join(dataset.model_path, 'seq_scatter.mp4'),
                                        cv2.VideoWriter_fourcc(*'mp4v'), 60,
                                        (ref_view_cam.image_width, ref_view_cam.image_height))
            for idx in tqdm(range(len(scatter_list))):
                scatter_np = scatter_list[idx]
                vout_scatter.write(scatter_np)

            vout_scatter.release()


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--inter", action="store_true", default=False)
    parser.add_argument("--begin_longi", "-b", default=45, type=int)
    parser.add_argument("--components", "-c", action="store_true", default=False)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    with torch.no_grad():
        render_sequence(model.extract(args), args.iteration, pipeline.extract(args), args.inter, args.begin_longi, args.components)



