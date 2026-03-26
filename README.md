
# `SceneTok`: A Compressed, Diffusable Token Space for 3D Scenes [CVPR '26] 
<a href="https://mohammadasim98.github.io">Mohammad Asim</a>, <a href="https://geometric-rl.mpi-inf.mpg.de/people/Wewer.html">Christopher Wewer</a>, <a href="https://geometric-rl.mpi-inf.mpg.de/people/lenssen.html">Jan Eric Lenssen</a>

*Max Planck Institute for Informatics, Saarland Informatics Campus*

[![Website](https://img.shields.io/badge/Website-SceneTok-blue)](https://geometric-rl.mpi-inf.mpg.de/scenetok/) [![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2602.18882)


### `TL;DR: A scene autoencoder that encodes a scene into a compressed, unstructured and diffusable 1D token representation.` 

### 📣 News

- **24.03.2026** - `SceneGen` Code, documentation and checkpoint release
- **09.03.2026** - `SceneTok` Code, documentation and checkpoint release
- **23.02.2026** - Initial repository release
- **21.02.2026** - Accepted to [`CVPR 2026`](https://cvpr.thecvf.com/) 🎉!

## 🔍 Method Overview 
<div align="center">
  <img src="assets/imgs/scenetok_method.svg" width="800"/>
  
</div>
<div align="justify">
<p>
  
  <b>(Left)</b> `SceneTok` encodes view sets into a set of compressed, unstructured scene tokens by chaining a VA-VAE image compressor and a perceiver module. The tokens can be rendered from novel views with a generative decoder based on rectified flows. 
  
  <b>(Right)</b> `SceneGen` perform scene generation by generating compressed scene tokens conditioned on a single or a few images and a set of anchor poses, defining the spatial scene extent.
</p>
</div>


## :wrench: Installation
### System Requirements
```bash
CUDA >= 11.8
Python >= 3.9
PyTorch >= 2.5
```
> [!NOTE]
> We only tested on the following so far: 
>
> `torch-2.5.1 + Python-3.11 + CUDA 12.1`
>
> `torch 2.7.0 + Python 3.11 + CUDA 12.8`
<details>
  <summary>Hardware Requirements</summary>

Tested hardware for inference:

| GPU | Mem 
|-----------|-----------|
| L40S | 45GB |
| Quadro RTX 8000 | 48GB |
| RTX 4090 | 24GB |

Test hardware for training:

| GPU | Mem 
|-----------|-----------|
| A100 | 40GB/80GB |
| H100 | 96GB |

</details>
<details>
  <summary>Setting Up Environment</summary>
Clone the repository

```bash
git clone https://github.com/mohammadasim98/scenetok.git --recursive
cd scenetok
```

You use any environment manager, we show an example for `conda`
```bash

conda create -n scenetok -python=3.11
conda activate scenetok

# torch 2.5.1 + cuda 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

```


Install remaining dependencies.
```bash
pip install -r requirements.txt
```

We use `Flash Attention 2` in our project, refer to their repository for more details regarding installation.

```bash
pip install psutils ninja packaging
pip install flash-attn --no-build-isolation
```

</details>
<details>
  <summary>Downloading Data</summary>

> #### **Datasets**
> Place the dataset in `dataset/` folder.
> Refer to [DATASET.md](/docs/DATASET.md) for more information into downloading and preprocessing the datasets. 


> #### **Models**
> Download the SceneTok models to `checkpoints/` folder.
> | Dataset | Model | rFID 
> |-----------|-----------|-----------|
> | RealEstate10K | [va-videodc_re10k.ckpt](https://nextcloud.mpi-klsb.mpg.de/index.php/s/6Y7EsosfbnpcRxj)   | 11.12   |
> | DL3DV | [va-videodc_dl3dv.ckpt](https://nextcloud.mpi-klsb.mpg.de/index.php/s/aYBX7atFNKkmdSE)   | 19.12   |
> | | [va-wan_dl3dv.ckpt](https://nextcloud.mpi-klsb.mpg.de/index.php/s/X7yzk7QANtwawPc)   | **14.30**   |

> Download the SceneGen models to `checkpoints/` folder. Note that you also need to download the corresponding [va-videodc_re10k_scene.ckpt](https://nextcloud.mpi-klsb.mpg.de/index.php/s/zEsw8ttT9E6Ge3C) model used for the SceneGen.
> | Dataset | Model | gFID 
> |-----------|-----------|-----------|
> | RealEstate10K | [scenegen_shift1_re10k.ckpt](https://nextcloud.mpi-klsb.mpg.de/index.php/s/3XrHrKQR8diodAa)   | 19.99   |
> |  | [scenegen_shift4_re10k.ckpt](https://nextcloud.mpi-klsb.mpg.de/index.php/s/GNbPMdd6gWtM56k)   | 19.04   |
> |  | [scenegen_shift12_re10k.ckpt](https://nextcloud.mpi-klsb.mpg.de/index.php/s/oDDgXYwDXijQo6a)   | 18.90   |

> #### **Pretrained AEs**
> 
> Download [VA-VAE](https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/blob/main/vavae-imagenet256-f16d32-dinov2.pt) including the [latent statistics](https://huggingface.co/hustvl/vavae-imagenet256-f16d32-dinov2/blob/main/latents_stats.pt) to `checkpoints/` folder.
> 
> Download [VideoDCAE](https://huggingface.co/hpcai-tech/Open-Sora-v2-Video-DC-AE/resolve/main/F32T4C128_AE.safetensors) to `checkpoint/` folder.
>
> Download [Wan 2.2 VAE](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/resolve/main/Wan2.2_VAE.pth) to `checkpoint/` folder.

</details>

> [!NOTE]
> If you are using other environments than `conda`, then modify the environmental setup in `scripts/*sh`.

## :running: Quick Inference in Jupyter Notebook

We provide an experimental notebook to perform quick inference with [SceneTok](/notebook/scenetok.ipynb) and [SceneGen](/notebook/scenegen.ipynb). 

## :robot: Training SceneTok
Run the following to initiate training:
```bash
python -m src.main +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  wandb.activated=true
```
> [!IMPORTANT]
> In case of memory issues (OOM), you can reduce the overal memory consumption by setting `gradient_checkpointing=true` for `model.denoiser` and `model.compressor` in `config/experiments/*.yaml`. They are set to `false`  by default.
> set environment variable `DEBUG=1` for debugging purposes which disables `torch.compile` for all modules. Note that `torch.compile` is enabled by default for all modules (only used during training step).
Set the parameters according to your needs. We provide the following configurations files `${config}` for `SceneTok`. Note that you can modifiy/add your own configurations. The configuration for DL3DV finetunes pretrained weights from RealEstate10K, but you can train from scratch as well.
```bash
# RealEstate10
scenetok_va-vdc_lognorm_re10k_scratch

# DL3DV (VA-VAE + VideoDCAE)
scenetok_va-vdc_shift4_dl3dv_finetuned
scenetok_va-vdc_shift8_dl3dv_finetuned

# DL3DV (VA-VAE + Wan)
scenetok_va-wan_shift4_dl3dv_finetuned
scenetok_va-wan_shift8_dl3dv_finetuned
```
<details>
  <summary>Using SLURM</summary>

  We provide SLURM script to allow multi-node/gpu training using job arrays. In `scripts/slurm_job_array_scenetok.sh` we provide a template where you can define your own configuration related to GPU and job array configurations. 
  First, set the `PROJECT_ROOT` as environment variable:
  ```bash
  export PROJECT_ROOT="<your-project-directory>"
  ```
  The simply run it as:

  ```bash
  bash slurm_job_array_scenetok.sh ${config}
  ```  
</details>


## :gear: Inference on SceneTok

```bash
python -m src.main +experiment=${config} mode=test hydra.job.name=test \
  dataset=${dataset} \ 
  wandb.activated=false \ 
  trainer.limit_test_batches=1 \
  data_loader.test.batch_size=1 \
  model.compressor.ckpt_path=${ckpt} \
  model.compressor.load_strict=false \
  model.denoiser.ckpt_path=${ckpt} \
  model.denoiser.load_strict=false \
  dataset.root=${data_root} \
  hydra.run.dir=${output_dir} \
  dataset/view_sampler=${view_sampler} \
  dataset.view_sampler.index_path=${index_path}
```
> [!NOTE]
> `model.compressor` and `model.denoiser` uses the same checkpoint file
> `load_strict` should be set to `false` to allow backward compatibility for all modules.

Select ```view_sampler``` from `evaluation_video` for VideoDCAE-based models and `evaluation_video_wan` for Wan-based models.

List below are the `config` with checkpoints weights provided [here](#models)
```bash
# RealEstate10
scenetok_va-vdc_lognorm_re10k_scratch # va-videodc_re10k.ckpt

# DL3DV (VA-VAE + VideoDCAE)
scenetok_va-vdc_shift8_dl3dv_finetuned # va-videodc_dl3dv.ckpt

# DL3DV (VA-VAE + Wan)
scenetok_va-wan_shift4_dl3dv_finetuned # va-wan_dl3dv.ckpt
```
`index_path` are view indices to use for context and targets. We provide several index files in `assets/evaluation_index/`

Choose `dataset` from `re10k` and `dl3dv` for RealEstate10K and DL3DV datasets respectively.

## :robot: Training SceneGen
SceneGen generates compressed scene tokens conditioned on a single or a few images and a set of anchor poses. It requires a pretrained SceneTok model.

Run the following to initiate training:
```bash
python -m src.main_scene +experiment=${config} \
  data_loader.train.num_workers=${num_workers} \
  mode=train \
  trainer.devices=${gpus} \
  trainer.num_nodes=${num_nodes} \
  wandb.activated=true
```

> [!IMPORTANT]
> SceneGen uses `src.main_scene` instead of `src.main` as the entry point.

Set the parameters according to your needs. We provide the following configurations files `${config}` for `SceneGen`:
```bash
# RealEstate10K
scenegen_shift1_re10k   # timestep_shift=1
scenegen_shift4_re10k   # timestep_shift=4
scenegen_shift12_re10k  # timestep_shift=12
```

> [!NOTE]
> The `timestep_shift` parameter controls the noise schedule. Higher shift values generally lead to better generation quality. The pretrained SceneTok model `va-videodc_re10k_scene.ckpt` is required and automatically loaded via the config.

<details>
  <summary>Using SLURM</summary>

  We provide SLURM script to allow multi-node/gpu training using job arrays. In `scripts/slurm_job_array_scenegen.sh` we provide a template where you can define your own configuration related to GPU and job array configurations. 
  First, set the `PROJECT_ROOT` as environment variable:
  ```bash
  export PROJECT_ROOT="<your-project-directory>"
  ```
  Then simply run it as:

  ```bash
  bash scripts/slurm_job_array_scenegen.sh ${config}
  ```  
</details>


## :gear: Inference on SceneGen

```bash
python -m src.main_scene +experiment=${config} mode=test hydra.job.name=test \
  dataset=re10k \
  wandb.activated=false \
  trainer.limit_test_batches=1 \
  data_loader.test.batch_size=1 \
  model.compressor.ckpt_path=${scenetok_ckpt} \
  model.compressor.load_strict=false \
  model.denoiser.ckpt_path=${scenetok_ckpt} \
  model.denoiser.load_strict=false \
  model.scene_generator.ckpt_path=${scenegen_ckpt} \
  model.scene_generator.load_strict=false \
  dataset.root=${data_root} \
  hydra.run.dir=${output_dir} \
  dataset/view_sampler=evaluation_video \
  dataset.view_sampler.index_path=${index_path}
```
> [!NOTE]
> `model.compressor` and `model.denoiser` use the same SceneTok checkpoint (`va-videodc_re10k_scene.ckpt`), while `model.scene_generator` uses a separate SceneGen checkpoint.
> `load_strict` should be set to `false` to allow backward compatibility for all modules. 

List below are the `config` and checkpoint combinations:
```bash
# RealEstate10K
scenegen_shift1_re10k   # scenegen_shift1_re10k.ckpt
scenegen_shift4_re10k   # scenegen_shift4_re10k.ckpt
scenegen_shift12_re10k  # scenegen_shift12_re10k.ckpt

# All SceneGen configs require:
# - scenetok_ckpt=checkpoints/va-videodc_re10k_scene.ckpt
# - scenegen_ckpt=checkpoints/scenegen_shift{1,4,12}_re10k.ckpt
```

We also provide a Jupyter notebook for interactive inference: [SceneGen Notebook](/notebook/scenegen.ipynb)


## ✅ TODO
`SceneTok`
- [x] RealEstate10K - VA-VAE+VideoDC
- [x] DL3DV - VA-VAE+Wan
- [x] DL3DV - VA-VAE+VideoDC

`SceneGen`
- [x] RealEstate10K - VA-VAE+VideoDC

## :mega: Future Extensions
- [ ] Interactive Scene Renderer
  - [ ] Causal decoder version for SceneTok
- [ ] High-resolution SceneTok (512x512)
- [ ] Pointcloud prediction from RGB+Depth renderings
- [ ] SceneGen (DL3DV)

## :page_with_curl: Feedback
We will be very happy to hear feedback on how best to improve the documentation. Since we cleaned and restructured the code alot, in case of any issue or bugs with the documentation or the codebase, feel free to let us know. Refer to [KNOWN_BUGS](/docs/KNOWN_BUGS.md) for bugs already reported.

## :bookmark: BibTeX
If you find our work useful in your research, please cite it as:
<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <pre><code>@inproceedings{asim26scenetok,
    title = {SceneTok: A Compressed, Diffusable Token Space for 3D Scenes},
    author = {Asim, Mohammad and Wewer, Christopher and Lenssen, Jan Eric},
    booktitle = {IEEE/CVF Computer Vision and Pattern Recognition ({CVPR})},
    year = {2026},
}</code></pre>
  </div>
</section>



