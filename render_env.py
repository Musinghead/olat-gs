import torch
import torch.nn.functional as F
from scene import Scene
import os
from tqdm import tqdm
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel, query_olat, rasterize_color
from scene.cameras import Camera
from PIL import Image
import numpy as np
import cv2
from time import time

def read_hdri(fpath):
    with open(fpath, "rb") as h:
        buffer_ = np.frombuffer(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).clip(0, 1)
    # linear space
    return rgb

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


def render_env(dataset : ModelParams, iteration : int, pipeline : PipelineParams, hdr_fpath, cam_longi=0, inten_factor=1., env_longi_offset=0.):
    # read an env map, as torch tensor
    hdr_map = torch.tensor(read_hdri(hdr_fpath), dtype=torch.float32, device='cuda')
    hdr_height, hdr_width = hdr_map.shape[0], hdr_map.shape[1]

    # load a trained model
    dataset.eval = True
    gaussians = GaussianModel(dataset.sh_degree)

    # add an option to not read images
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_train_img=True, skip_test_img=True)

    n_chan = 6
    bg_color = [1 for _ in range(n_chan)] if dataset.white_background else [0 for _ in range(n_chan)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # get average cam and pl distance for reference
    train_cameras = scene.getTrainCameras()
    ref_view_cam = train_cameras[0]

    down = torch.tensor([[0., -1., 0.]], dtype=torch.float32, device='cuda')
    # avg distance
    cam_radius = torch.stack([cam.camera_center for cam in train_cameras], dim=0).norm(dim=-1).mean() * 1.
    pl_radius = torch.stack([cam.pl_pos for cam in train_cameras], dim=0).norm(dim=-1).mean() * 1.

    print('cam radius', cam_radius)
    print('pl radius', pl_radius)

    # generate a sample view, similar to render_seq
    cam_lati = torch.tensor([45.], dtype=torch.float32, device='cuda')
    cam_longi = torch.tensor(cam_longi, dtype=torch.float32, device='cuda')

    cam_pos = spher2cart(cam_radius, cam_lati, cam_longi)  # 1 x 3
    cam_c2w = get_cam_view_mat(cam_pos, down)
    cam_pose_w2c = torch.linalg.inv(cam_c2w)
    R = cam_pose_w2c[:3, :3].T
    T = cam_pose_w2c[:3, 3]

    # determine number of pixel samples
    lgt_sample_longs = (torch.arange(0, 360, 10, dtype=torch.long, device='cuda') + env_longi_offset) % 360
    lgt_sample_lats = torch.arange(20, 90, 10, dtype=torch.long, device='cuda')
    olat_accum_color = torch.zeros_like(gaussians.get_xyz)
    num_samples = len(lgt_sample_longs) * len(lgt_sample_lats)
    print('{} longi x {} lati = {} samples'.format(len(lgt_sample_longs), len(lgt_sample_lats), num_samples))

    render_begin = time()

    for longi_idx, lgt_sample_longi in tqdm(enumerate(lgt_sample_longs)):
        for lgt_sample_lati in lgt_sample_lats:

            pl_pos = spher2cart(pl_radius, lgt_sample_lati, lgt_sample_longi).cpu().numpy()
            # sample pixels on the env map, query olats and average as the color
            lgt_longi_idx = (lgt_sample_longi / 360 * hdr_width).to(dtype=torch.long)
            lgt_lati_idx = ((90 - lgt_sample_lati) / 180 * hdr_height).to(dtype=torch.long)

            pixel_color = hdr_map[lgt_lati_idx, lgt_longi_idx].reshape(1, 3)

            sample_cam = Camera(colmap_id=0, R=R.cpu().numpy(), T=T.cpu().numpy(), FoVx=ref_view_cam.FoVx,
                                FoVy=ref_view_cam.FoVy, cx=ref_view_cam.cx, cy=ref_view_cam.cy,
                                pl_pos=pl_pos, image=None, width=ref_view_cam.image_width, height=ref_view_cam.image_height,
                                gt_alpha_mask=None, image_name=None, uid=0)

            lgt_depth = scene.render_lgt_depth(sample_cam)

            olat_value = query_olat(sample_cam, gaussians, lgt_depth) * pixel_color
            olat_accum_color += olat_value

    olat_avg_color = olat_accum_color / num_samples
    olat_final_color = olat_avg_color * inten_factor

    # rasterize and gamma correction
    sample_cam = Camera(colmap_id=0, R=R.cpu().numpy(), T=T.cpu().numpy(), FoVx=ref_view_cam.FoVx,
                        FoVy=ref_view_cam.FoVy, cx=ref_view_cam.cx, cy=ref_view_cam.cy,
                        pl_pos=np.array([[1, 1, 1]], dtype=np.float32), image=None, width=ref_view_cam.image_width, height=ref_view_cam.image_height,
                        gt_alpha_mask=None, image_name=None, uid=0)
    rendered_image = rasterize_color(sample_cam, gaussians, olat_final_color, pipeline, background).permute(1,2,0).cpu().numpy()
    rendered_image = (rendered_image * 255).clip(0, 255).astype(np.uint8)

    render_end = time()

    print('total render time used {:.4f}'.format(render_end - render_begin))

    # save
    save_fdr = os.path.join(args.model_path, 'env_relit')
    os.makedirs(save_fdr, exist_ok=True)
    hdr_name = os.path.split(hdr_fpath)[-1][:-4]
    Image.fromarray(rendered_image, 'RGB').save(os.path.join(save_fdr, '{}_{}_{:.1f}.png'.format(hdr_name, cam_longi, inten_factor)))


def render_env_seq(dataset: ModelParams, iteration: int, pipeline: PipelineParams, hdr_fpath, begin_cam_longi=45, inten_factor=1.):
    # read an env map, as torch tensor
    hdr_map = torch.tensor(read_hdri(hdr_fpath), dtype=torch.float32, device='cuda')
    hdr_height, hdr_width = hdr_map.shape[0], hdr_map.shape[1]

    # load a trained model
    dataset.eval = True
    gaussians = GaussianModel(dataset.sh_degree)

    # add an option to not read images
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_train_img=True, skip_test_img=True)

    n_chan = 6
    bg_color = [1 for _ in range(n_chan)] if dataset.white_background else [0 for _ in range(n_chan)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # get average cam and pl distance for reference
    train_cameras = scene.getTrainCameras()
    ref_view_cam = train_cameras[0]

    down = torch.tensor([[0., -1., 0.]], dtype=torch.float32, device='cuda')
    # avg distance
    cam_radius = torch.stack([cam.camera_center for cam in train_cameras], dim=0).norm(dim=-1).mean() * 1.
    pl_radius = torch.stack([cam.pl_pos for cam in train_cameras], dim=0).norm(dim=-1).mean() * 1.

    print('cam radius', cam_radius)
    print('pl radius', pl_radius)

    # save
    hdr_name = os.path.split(hdr_fpath)[-1][:-4]
    save_fdr = os.path.join(args.model_path, 'env_relit', hdr_name)
    os.makedirs(save_fdr, exist_ok=True)

    # seq 1, fix cam, rotate env map
    cam_lati = torch.tensor([45.], dtype=torch.float32, device='cuda')
    cam_longi = torch.tensor([begin_cam_longi], dtype=torch.float32, device='cuda')

    cam_pos = spher2cart(cam_radius, cam_lati, cam_longi)  # 1 x 3
    cam_c2w = get_cam_view_mat(cam_pos, down)
    cam_pose_w2c = torch.linalg.inv(cam_c2w)
    R = cam_pose_w2c[:3, :3].T
    T = cam_pose_w2c[:3, 3]

    seq1_list = []
    # determine number of pixel samples
    lgt_sample_lats = torch.arange(20, 90, 10, dtype=torch.long, device='cuda')
    lgt_sample_longs = torch.arange(0, 360, 10, dtype=torch.long, device='cuda')
    num_samples = len(lgt_sample_longs) * len(lgt_sample_lats)

    for fid, env_longi_offset in tqdm(enumerate(range(0, 360, 2))):
        olat_accum_color = torch.zeros_like(gaussians.get_xyz)

        for longi_idx, lgt_sample_longi in enumerate(lgt_sample_longs):
            for lgt_sample_lati in lgt_sample_lats:
                pl_pos = spher2cart(pl_radius, lgt_sample_lati, lgt_sample_longi).cpu().numpy()
                # sample pixels on the env map, query olats and average as the color
                lgt_shift_longi = (lgt_sample_longi + env_longi_offset) % 360
                lgt_longi_idx = (lgt_shift_longi / 360 * hdr_width).to(dtype=torch.long)
                lgt_lati_idx = ((90 - lgt_sample_lati) / 180 * hdr_height).to(dtype=torch.long)

                pixel_color = hdr_map[lgt_lati_idx, lgt_longi_idx].reshape(1, 3)

                sample_cam = Camera(colmap_id=0, R=R.cpu().numpy(), T=T.cpu().numpy(), FoVx=ref_view_cam.FoVx,
                                    FoVy=ref_view_cam.FoVy, cx=ref_view_cam.cx, cy=ref_view_cam.cy,
                                    pl_pos=pl_pos, image=None, width=ref_view_cam.image_width,
                                    height=ref_view_cam.image_height,
                                    gt_alpha_mask=None, image_name=None, uid=0)

                lgt_depth = scene.render_lgt_depth(sample_cam)

                olat_value = query_olat(sample_cam, gaussians, lgt_depth) * pixel_color
                olat_accum_color += olat_value

        olat_avg_color = olat_accum_color / num_samples
        olat_final_color = olat_avg_color * inten_factor

        # rasterize and gamma correction
        sample_cam = Camera(colmap_id=0, R=R.cpu().numpy(), T=T.cpu().numpy(), FoVx=ref_view_cam.FoVx,
                            FoVy=ref_view_cam.FoVy, cx=ref_view_cam.cx, cy=ref_view_cam.cy,
                            pl_pos=np.array([[1, 1, 1]], dtype=np.float32), image=None, width=ref_view_cam.image_width,
                            height=ref_view_cam.image_height,
                            gt_alpha_mask=None, image_name=None, uid=0)
        rendered_image = rasterize_color(sample_cam, gaussians, olat_final_color, pipeline, background).permute(1, 2,
                                                                                                                0).cpu().numpy()
        rendered_image = (rendered_image * 255).clip(0, 255).astype(np.uint8)

        rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)
        seq1_list.append(rendered_image)

        cv2.imwrite(os.path.join(save_fdr, 'seq1_l{}_i{:.1f}_f{}.png'.format(int(cam_longi.cpu().item()), inten_factor, fid)),
                    rendered_image)

    # seq 2, fix env map, rotate view
    seq2_list = []
    # generate a sample view, similar to render_seq
    cam_lati = torch.tensor([45.], dtype=torch.float32, device='cuda')
    # determine number of pixel samples
    lgt_sample_longs = (torch.arange(0, 360, 10, dtype=torch.long, device='cuda')) % 360
    lgt_sample_lats = torch.arange(20, 90, 10, dtype=torch.long, device='cuda')
    num_samples = len(lgt_sample_longs) * len(lgt_sample_lats)
    for fid, cam_longi in tqdm(enumerate(range(begin_cam_longi, 360 + begin_cam_longi, 2))):
        cam_longi = torch.tensor(cam_longi, dtype=torch.float32, device='cuda')

        cam_pos = spher2cart(cam_radius, cam_lati, cam_longi)  # 1 x 3
        cam_c2w = get_cam_view_mat(cam_pos, down)
        cam_pose_w2c = torch.linalg.inv(cam_c2w)
        R = cam_pose_w2c[:3, :3].T
        T = cam_pose_w2c[:3, 3]

        olat_accum_color = torch.zeros_like(gaussians.get_xyz)

        for longi_idx, lgt_sample_longi in enumerate(lgt_sample_longs):
            for lgt_sample_lati in lgt_sample_lats:
                pl_pos = spher2cart(pl_radius, lgt_sample_lati, lgt_sample_longi).cpu().numpy()
                # sample pixels on the env map, query olats and average as the color
                lgt_longi_idx = (lgt_sample_longi / 360 * hdr_width).to(dtype=torch.long)
                lgt_lati_idx = ((90 - lgt_sample_lati) / 180 * hdr_height).to(dtype=torch.long)

                pixel_color = hdr_map[lgt_lati_idx, lgt_longi_idx].reshape(1, 3)

                sample_cam = Camera(colmap_id=0, R=R.cpu().numpy(), T=T.cpu().numpy(), FoVx=ref_view_cam.FoVx,
                                    FoVy=ref_view_cam.FoVy, cx=ref_view_cam.cx, cy=ref_view_cam.cy,
                                    pl_pos=pl_pos, image=None, width=ref_view_cam.image_width,
                                    height=ref_view_cam.image_height,
                                    gt_alpha_mask=None, image_name=None, uid=0)

                lgt_depth = scene.render_lgt_depth(sample_cam)

                olat_value = query_olat(sample_cam, gaussians, lgt_depth) * pixel_color
                olat_accum_color += olat_value

        olat_avg_color = olat_accum_color / num_samples
        olat_final_color = olat_avg_color * inten_factor

        # rasterize and gamma correction
        sample_cam = Camera(colmap_id=0, R=R.cpu().numpy(), T=T.cpu().numpy(), FoVx=ref_view_cam.FoVx,
                            FoVy=ref_view_cam.FoVy, cx=ref_view_cam.cx, cy=ref_view_cam.cy,
                            pl_pos=np.array([[1, 1, 1]], dtype=np.float32), image=None, width=ref_view_cam.image_width,
                            height=ref_view_cam.image_height,
                            gt_alpha_mask=None, image_name=None, uid=0)
        rendered_image = rasterize_color(sample_cam, gaussians, olat_final_color, pipeline, background).permute(1, 2, 0).cpu().numpy()
        rendered_image = (rendered_image * 255).clip(0, 255).astype(np.uint8)

        rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)
        seq2_list.append(rendered_image)

        cv2.imwrite(os.path.join(save_fdr, 'seq2_l{}_i{:.1f}_f{}.png'.format(int(cam_longi.cpu().item()), inten_factor, fid)),
                    rendered_image)

    vout = cv2.VideoWriter(os.path.join(dataset.model_path, 'env_relit.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 60,
                           (ref_view_cam.image_width, ref_view_cam.image_height))
    # os.makedirs(save_folder, exist_ok=True)
    for idx in tqdm(range(len(seq1_list))):
        image_np = seq1_list[idx]
        vout.write(image_np)


    for idx in tqdm(range(len(seq2_list))):
        image_np = seq2_list[idx]
        vout.write(image_np)

    vout.release()


if __name__ == '__main__':
    # cfg
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--hdr_fpath", type=str, required=True)
    parser.add_argument("--longi", default=45, type=int)
    parser.add_argument("--inten", default=4., type=float)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    with torch.no_grad():
        # render_env(model.extract(args), args.iteration, pipeline.extract(args), args.hdr_fpath, args.longi, args.inten)
        render_env_seq(model.extract(args), args.iteration, pipeline.extract(args), args.hdr_fpath, args.longi, args.inten)