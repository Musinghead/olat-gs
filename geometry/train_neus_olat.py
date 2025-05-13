"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import pathlib
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets.olat import SubjectLoader
from radiance_fields.field import SDFNetwork, RenderingNetworkPl, SingleVarianceNetwork

import os
import trimesh
import shutil

from utils import (
    render_rays_with_occgrid_neus_olat,
    render_image_with_occgrid_neus_olat,
    set_random_seed,
    extract_geometry
)
from nerfacc.estimators.occ_grid import OccGridEstimator

device = "cuda"
set_random_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    "-r",
    type=str,
    default=str("../data/olat"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="the path of the pretrained model",
)
parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    help="save path of results",
)
parser.add_argument(
    "--key",
    "-k",
    type=str,
    default=None,
    help="key tag reminder",
)
parser.add_argument(
    "--scene",
    "-s",
    type=str,
    default="Pixiu",
    help="which scene to use",
)
parser.add_argument(
    "--test_chunk_size",
    type=int,
    default=4096,
)
parser.add_argument(
    '--camopt', 
    action='store_true', 
    default=False
)
parser.add_argument(
    "--sdf_scale",
    type=float,
    default=3.0
)
parser.add_argument(
    "--bslmt",
    type=int,
    default=None
)
parser.add_argument(
    "--camopt_start",
    type=int,
    default=0
)
parser.add_argument(
    "--camopt_lr",
    type=float,
    default=3e-5
)
parser.add_argument(
    "--womask_end",
    type=int,
    default=-1
)
parser.add_argument(
    "--end_iter",
    type=int,
    default=30_000
)
args = parser.parse_args()

if args.save_path is None:
    args.save_path = os.path.join("output", args.scene, (f"{args.key}" if args.key else "example"))
os.makedirs(args.save_path, exist_ok=True)

# backup code files
print(f'> record code files')
record_fdr = os.path.join(args.save_path, 'record')
os.makedirs(record_fdr, exist_ok=True)
shutil.copyfile(f"./train_neus_olat.py", os.path.join(record_fdr, f"train_neus_olat.py"))
shutil.copyfile(f"./utils.py", os.path.join(record_fdr, f"utils.py"))
shutil.copyfile(f"./datasets/olat.py", os.path.join(record_fdr, f"olat.py"))
shutil.copyfile(f"./radiance_fields/field.py", os.path.join(record_fdr, f"field.py"))


train_conf = {
    'learning_rate': 5e-4,
    'camopt_lr': args.camopt_lr,
    'camopt_start': args.camopt_start,
    'learning_rate_alpha': 0.05,
    'end_iter': args.end_iter,

    'batch_size': 512,
    # 'validate_resolution_level': 4,
    'warm_up_end': 5000,
    'anneal_end': 0,
    'use_white_bkgd': False,

    'save_freq': 10000,
    'val_freq': 10_000,
    'val_mesh_freq': 10_000,
    # 'report_freq': 100,

    'igr_weight': 0.1,
    'mask_weight': 0.1,
    'womask_end': args.womask_end # omit mask loss at beginning for a smooth warm-up
}

if args.camopt:
    print(f"> start camopt from iter {train_conf['camopt_start']}")

# training parameters
max_steps = train_conf['end_iter']
init_batch_size = 512
# max_batch_size = args.bs
target_sample_batch_size = 1 << 16
# scene parameters
aabb = torch.tensor([-1., -1., -1., 1., 1., 1.], device=device)
near_plane = 0.0
far_plane = 1.0e10
# model parameters
grid_resolution = 128
grid_nlvl = 1
# render parameters
render_step_size = 5e-3

# setup the dataset
train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=init_batch_size,
    device=device,
    camopt=args.camopt,
)

# test dataset, TODO only load 2 views to visualize
test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    device=device,
)

estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)

# network conf
sdf_net_conf =  {
    'd_out': 257,
    'd_in': 3,
    'd_hidden': 256,
    'n_layers': 8,
    'skip_in': [4],
    'multires': 6,
    'bias': 0.5,
    'scale': args.sdf_scale,
    'geometric_init': True,
    'weight_norm': True
}

variance_net_conf = {
    'init_val': 0.3
}

render_net_conf = {
    'd_feature': 256,
    'mode': 'idr',
    'd_in': 12,
    'd_out': 3,
    'd_hidden': 256,
    'n_layers': 4,
    'weight_norm': True,
    'multires_view': 4,
    'squeeze_out': True
}

# setup the NeuS radiance field we want to train.
sdf_net = SDFNetwork(**sdf_net_conf).to(device)
render_net = RenderingNetworkPl(**render_net_conf).to(device)
variance_net = SingleVarianceNetwork(**variance_net_conf).to(device)

# optimizer
l = [
    {'params': sdf_net.parameters(), 'lr': train_conf['learning_rate'], 'base_lr': train_conf['learning_rate'], 'name': 'sdf'},
    {'params': render_net.parameters(), 'lr': train_conf['learning_rate'], 'base_lr': train_conf['learning_rate'], 'name': 'render'},
    {'params': variance_net.parameters(), 'lr': train_conf['learning_rate'], 'base_lr': train_conf['learning_rate'], 'name': 's_val'},
    {'params': train_dataset.cam_pose_resi.parameters(), 'lr': train_conf['camopt_lr'], 'base_lr': train_conf['camopt_lr'], 'name': 'camopt'}
]
optimizer = torch.optim.Adam(l)

# freeze cam pose resi at beginning
for param in train_dataset.cam_pose_resi.parameters():
    param.requires_grad_(False)

# vanilla NeuS' lr scheduler
def update_learning_rate(iter_step):
    if iter_step < train_conf['warm_up_end']:
        learning_factor = iter_step / train_conf['warm_up_end']
    else:
        alpha = train_conf['learning_rate_alpha']
        progress = (iter_step - train_conf['warm_up_end']) / (train_conf['end_iter'] - train_conf['warm_up_end'])
        learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

    for g in optimizer.param_groups:
        g['lr'] = g['base_lr'] * learning_factor

def get_cos_anneal_ratio(iter_step):
    if train_conf['anneal_end'] == 0.0:
        return 1.0
    else:
        return np.min([1.0, iter_step / train_conf['anneal_end']])

step = 0
if args.model_path is not None:
    checkpoint = torch.load(args.model_path)

    sdf_net.load_state_dict(checkpoint["sdf_net_state_dict"])
    render_net.load_state_dict(checkpoint["render_net_state_dict"])
    variance_net.load_state_dict(checkpoint["render_net_state_dict"])
    train_dataset.cam_pose_resi.load_state_dict(checkpoint["cam_pose_resi"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    estimator.load_state_dict(checkpoint["estimator_state_dict"])
    step = checkpoint["step"]
else:
    step = 0

# training
tic = time.time()

# tqdm bar
pbar = tqdm(range(max_steps + 1), dynamic_ncols=True)

num_rays = init_batch_size

# for step in tqdm(range(max_steps + 1), dynamic_ncols=True):
for step in range(max_steps + 1):

    estimator.train()

    if args.camopt and step == train_conf['camopt_start']:
        print(f'step {step}, activate cam pose resi')
        for param in train_dataset.cam_pose_resi.parameters():
            param.requires_grad_(True)

    update_learning_rate(step)

    i = torch.randint(0, len(train_dataset), (1,)).item()

    data = train_dataset.fetch_data(i, args.camopt and step >= train_conf['camopt_start'])
    data = train_dataset.preprocess(data)

    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]
    mask = data["alpha"]
    pl_pos = data['pl_pos']

    # This implementation is borrowed from https://github.com/bennyguo/instant-nsr-pl
    def occ_eval_fn(x):
        sdf = sdf_net.sdf(x) # (n, 1)
        inv_s = variance_net(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        inv_s = inv_s.expand(sdf.shape[0], 1) # (n, 1)
        estimated_next_sdf = sdf - render_step_size * 0.5
        estimated_prev_sdf = sdf + render_step_size * 0.5
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
        return alpha # (n, 1)

    # update occupancy grid
    estimator.update_every_n_steps(
        step=step,
        occ_eval_fn=occ_eval_fn,
        occ_thre=1e-3
    )
    
    cos_anneal_ratio = get_cos_anneal_ratio(step)

    # render
    rgb, normal, n_rendering_samples, gradient_error, inv_s, opacity = render_rays_with_occgrid_neus_olat(
        sdf_net,
        render_net,
        variance_net,
        estimator,
        rays.origins,
        rays.viewdirs,
        pl_pos,
        # rendering options
        cos_anneal_ratio=cos_anneal_ratio,
        near_plane=near_plane,
        render_step_size=render_step_size,
        render_bkgd=None,
        is_training=True,
    )
    if n_rendering_samples == 0:
        continue

    if target_sample_batch_size > 0:
        # dynamic batch size for rays to keep sample batch size constant.
        num_rays = len(pixels)
        num_rays = int(
            num_rays * (target_sample_batch_size / float(n_rendering_samples))
        )

        if args.bslmt is not None:
            num_rays = min(num_rays, args.bslmt)
        train_dataset.update_num_rays(num_rays)

    mask = (mask > 0.5).float()

    mask_weight = train_conf['mask_weight'] if step > train_conf['womask_end'] else 0.0

    if mask_weight == 0.0:
        mask = torch.ones_like(mask)

    mask_sum = mask.sum() + 1e-5

    color_error = (rgb - pixels) * mask
    color_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
    mask_loss = F.binary_cross_entropy(opacity.clip(1e-3, 1.0 - 1e-3), mask)

    loss = color_loss + mask_loss * mask_weight + gradient_error * train_conf['igr_weight']

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step > 0 and step % 10 == 0:
        pbar.set_postfix({
            "loss": f"{loss.detach().cpu().item():.4f}", 
            "inv_s": f"{inv_s.detach().cpu().item():.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
            "n_ray": f"{len(pixels)}"
            })
        pbar.update(10)
 
    if step > 0 and (step % train_conf['val_freq'] == 0 or step == args.end_iter):
        # visualize rgb and normal
        with torch.no_grad():
            save_fdr = os.path.join(args.save_path, f'step_imgs', f'step_{step:06d}')
            os.makedirs(save_fdr, exist_ok=True)

            for i in range(0, len(test_dataset)):

                data = test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]
                pl_pos = data["pl_pos"]

                # rendering
                rgb, normal = render_image_with_occgrid_neus_olat(
                    sdf_net,
                    render_net,
                    variance_net,
                    estimator,
                    rays,
                    pl_pos,
                    # rendering options
                    cos_anneal_ratio=cos_anneal_ratio,
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=None,
                    test_chunk_size=min(num_rays, args.test_chunk_size),
                )
                imageio.imwrite(
                    os.path.join(save_fdr, f'rgb_{i}.png'),
                    (rgb.cpu().numpy() * 255).astype(np.uint8),
                )
                normal = normal * 0.5 + 0.5
                imageio.imwrite(
                    os.path.join(save_fdr, f'nrm_{i}.png'),
                    (normal.cpu().numpy() * 255).clip(0, 255).astype(np.uint8),
                )

    if step == args.end_iter:
        # extract and save mesh
        mesh_fdr = os.path.join(args.save_path, 'mesh')
        os.makedirs(mesh_fdr, exist_ok=True)
        resolution=512
        bound_min = torch.tensor([-1., -1., -1.], dtype=torch.float32, device=device)
        bound_max = torch.tensor([1., 1., 1.], dtype=torch.float32, device=device)
        vertices, triangles = extract_geometry(bound_min=bound_min, bound_max=bound_max, resolution=resolution, threshold=0.0, query_func=lambda pts: -sdf_net.sdf(pts))
        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(mesh_fdr, f"mesh_r{resolution}_iter{step:06d}.obj"))

    if step == args.end_iter:
        # save model
        ckpt_fdr = os.path.join(args.save_path, 'ckpt')
        os.makedirs(ckpt_fdr)
        model_save_path = os.path.join(ckpt_fdr, f"step_{step:06d}.pth")
        torch.save(
            {
                "step": step,
                "sdf_net_state_dict": sdf_net.state_dict(),
                "render_net_state_dict": render_net.state_dict(),
                "variance_net_state_dict": variance_net.state_dict(),
                "cam_posi_resi": train_dataset.cam_pose_resi.state_dict(),

                "optimizer_state_dict": optimizer.state_dict(),
                "estimator_state_dict": estimator.state_dict(),
            },
            model_save_path,
        )
