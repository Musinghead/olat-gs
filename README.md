# OLAT Gaussians for Generic Relightable Appearance Acquisition

[Website](https://musinghead.github.io/projects/olat_gaussians/) | [Data](https://drive.google.com/drive/folders/1Wm7kNqxENDFLtfVCp4AVM3K7JrHcvCWt?usp=sharing) | [Pretrained Models and Results](https://drive.google.com/drive/folders/1428s8xwEHb_dnry_Jd73xoWqwWLujsTZ?usp=sharing)

## Environment

```shell
conda create -n olat-gs
conda activate olat-gs
conda install pip python=3.10
pip install numpy==1.26 torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install plyfile tqdm trimesh opencv-python imageio scipy lpips PyMCubes matplotlib
pip install nerfacc -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-1.13.0_cu116.html
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

git clone https://github.com/Musinghead/olat-gs.git
cd olat-gs
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/nvdiffrast
```

**About tiny-cuda-nn**

1. The compilation of tinycudann requires gcc/g++ version >=8 and <=10, otherwise it may report errors. Before compiling tinycudann, specify gcc/g++ version if default version is not suitable:

```shell
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
```

2. On Windows, it may report an error,  "Cannot open include file: 'algorithm'". If so, follow the instructions [here](https://github.com/NVlabs/tiny-cuda-nn/issues/208#issuecomment-1858279465) to solve it.

## Data

Download olat data [here](https://drive.google.com/drive/folders/1Wm7kNqxENDFLtfVCp4AVM3K7JrHcvCWt?usp=sharing) and save under 'data' folder, so each case is saved under data/olat/<case_name>.

The test sets of case Cat, Fish, FurScene and Pixiu are from [DNL](https://yuedong.shading.me/project/defnlight/defnlight.htm).

The test sets of case CatSmall, CupFabric and Pikachu are from [NRHints](https://nrhints.github.io/).

## Run

### Train

**Proxy Mesh**

Pretrained proxy meshes are included in the data link for download and can be directly used to train OLAT Gaussians.

To generate a new proxy mesh:

```shell
cd geometry
python train_neus_olat.py --scene <case_name> --save_path <save_path> --camopt
cd ..
```

Refer to geometry/run.py for training commands for each case.

This is expected to work with masks (provided in the alpha channel of olat data) to get the best mesh quality.

To use the new proxy mesh, copy it to case folder:

```shell
cp geometry/<save_path>/mesh/mesh_r512_iter030000.obj data/olat/<case_name>/pmesh.obj
```

**OLAT Gaussians**

```shell
python train.py -s <case_folder> --eval [--key <key_tag>]
```

args:

--key: a key tag to be added to model_folder as a reminder

--no_refine_pose: disable refining camera pose if not required

### Render Test Set

```shell
python render.py -m <model_folder> --skip_train
```

args:

--no_refine_pose: disable refining camera pose if not required

### Evaluate Metrics

```shell
python metrics.py -m <model_folder>
```

### Run all cases to reproduce results

```shell
# run train, render, metric for all cases
python run.py --all --key verify
```

### Render Point Light Relighting Video

This can also be used to test rendering frame rate.

Render and save a video (part 1: fix camera, rotate light; part 2: rotate camera, fix light), named as model_folder/seq.mp4 by default.

```shell
python render_seq.py -m <model_folder> [-s <source_folder>]
```

Run interactive control (only tested on Windows).

```shell
python render_seq.py -m <model_folder> --inter [-s <source_folder>]
```

Args:

-s: specify source data folder when using a pretrained model.

### Render Environmental Relighting Video

Relighted frames are saved under model_folder/env_relit/envmap_name/. The relighted video is saved as model_folder/env_relit.mp4 (part 1: fix camera, rotate light; part 2: rotate camera, fix light).

Sample envmaps from [TensorIR](https://github.com/Haian-Jin/TensoIR?tab=readme-ov-file) are provided under envmaps/.

```shell
python render_env.py -m <model_folder> --hdr_fpath <path_to_envmap> [-s <source_folder>]
```

## Custom Data

We used OLAT data from [DNL](https://yuedong.shading.me/project/defnlight/defnlight.htm) and [NRHints](https://nrhints.github.io/). To capture new OLAT data, their data capture protocol is a great reference.

## Acknowledgement

This project is inspired and built on these great works:

[3DGS](https://github.com/graphdeco-inria/gaussian-splatting)

[Deferred Neural Lighting](https://yuedong.shading.me/project/defnlight/defnlight.htm)

[NRHints](https://nrhints.github.io/)

[Nerfacc](https://github.com/nerfstudio-project/nerfacc)

[Instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl)

[Tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)

[Instant-NGP](https://github.com/NVlabs/instant-ngp)

[NVDIFFRAST](https://github.com/NVlabs/nvdiffrast)

Other interesting relighting works:

[GS^3: Efficient Relighting with Triple Gaussian Splatting](https://gsrelight.github.io/)

[A Diffusion Approach to Radiance Field Relighting using Multi-Illumination Synthesis](https://repo-sam.inria.fr/fungraph/generative-radiance-field-relighting/)

## Citation

```
@inproceedings{kuang2024olat,
  title={OLAT Gaussians for Generic Relightable Appearance Acquisition},
  author={Kuang, Zhiyi and Yang, Yanchao and Dong, Siyan and Ma, Jiayue and Fu, Hongbo and Zheng, Youyi},
  booktitle={SIGGRAPH Asia 2024 Conference Papers},
  pages={1--11},
  year={2024}
}
```