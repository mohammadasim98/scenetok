
from pathlib import Path
from dataclasses import dataclass, field
from .autoencoder import AutoencoderCfg, AutoencodersCfg
from .denoiser import DenoiserCfg
from .scheduler import SchedulerCfg
from typing import Any, Dict, Literal, Union, List, Optional
from .compressor import CompressorCfg
from .scene_generator import SceneGeneratorCfg

@dataclass
class CameraEncoderCfg:
    num_origin_octaves: int=10
    num_direction_octaves: int=8


# @dataclass
# class SceneGeneratorCfg:

#     denoiser: DenoiserCfg
#     ckpt_path: Path | str

@dataclass
class CheckpointingCfg:
    denoiser_ckpt: str | Path | None=None
    compressor_ckpt: str | Path | None=None
    ignore_size_match: bool=False

@dataclass
class ModelCfg:
    
    denoiser: DenoiserCfg
    scheduler: SchedulerCfg
    compressor: CompressorCfg | None
    autoencoders:  AutoencodersCfg
    scene_scheduler: SchedulerCfg | None=None
    scene_generator: SceneGeneratorCfg | None=None
    use_cfg: bool=False
    cfg_scale: float=3.0
    scene_cfg_scale: float=3.0
    cfg_train: bool=True
    use_ray_encoding: bool=True
    srt_ray_encoding: bool=False
    use_ddim_scheduler: bool=False
    use_plucker: bool=False
    ema: bool=False
    use_ema_sampling: bool=False
    enable_xformers_memory_efficient_attention: bool=False
    only_target_mask: bool=True
    arbitrary_target_views: bool=False
    high_res_ray_map: bool=False
    high_res_ray_map_size: int=32
    enforce_uniform_noise: bool=True
    # backward compatibility for legacy experiments
    intr_scale: Union[tuple[float, float], list[float, float]] = (1.0, 1.0)
    normalize_ray_direction: bool=True
    switch_ray_order: bool=False
    force_clean: bool=False
    no_null_expand: bool=False
    noisy_scene_tokens: bool=False
    noise_prob: float=0.7
    mask_context: bool=False
    mask_tokens: bool=False
    random_masking: bool=False
    mu: float=-1.5
    sigma: float=1.5
    optimize_ddp: bool=False
    force_incorrect: bool=False
@dataclass  
class LRSchedulerCfg:
    name: str
    frequency: int = 1
    interval: Literal["epoch", "step"] = "step"
    kwargs: Dict[str, Any] | None = None



@dataclass
class FreezeCfg:
    denoiser: bool = False
    compressor: bool = False
    autoencoder: bool = True
    scene_generator: bool = False
@dataclass
class OptimizerCfg:
    name: str
    lr: float
    scale_lr: bool
    milestones: list[int] | None = None
    kwargs: Dict[str, Any] | None = None
    scheduler: LRSchedulerCfg | list[LRSchedulerCfg] | None = None
    override_lr: Optional[float] = None

@dataclass
class TestCfg:
    mode: Literal["sequential", "random", "interleave", "single", "pyramid", "pyramid_random", "autoregressive"] | None
    output_dir: str | Path
    scene_id: int | str | None = None
    generation_camera: bool = False
    window_shift: int = 1
    extrapolate_context: bool = False
    num_context_views: int = 12
    num_target_views: int = 8
    num_condition_views: int = 1
    reverse_sequence: bool = False
    scene_cfg: float=3.0
@dataclass
class ValCfg:
    mode: list[Literal["sequential", "random", "interleave", "single", "pyramid", "pyramid_random", "autoregressive"]]
    video: bool=False
    video_length: int=64
    window_shift: int = 1
    uncertainty_scale: float=5.2    
@dataclass
class TrainCfg:
    step_offset: int
    cfg_train: bool=False
    nan_skip: bool=True
    grad_norm_skip: bool=True
    grad_norm_skip_threshold: float=1.0