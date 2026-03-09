from typing import Literal, Optional, Tuple, List
from dataclasses import dataclass
from typing import Union
# Refer to https://github.com/huggingface/diffusers/blob/6131a93b969f87d171148bd367fd9990d5a49b6b/src/diffusers/models/autoencoders/autoencoder_kl.py#L38
@dataclass
class VideoDCKwargsCfg:
    
    in_channels: int = 3
    latent_channels: int = 128
    model_name="dc-ae-f32t4c128"
    from_scratch: bool = False
    is_training: bool = False
    use_spatial_tiling: bool = True
    use_temporal_tiling: bool = True
    spatial_tile_size: int = 256
    temporal_tile_size: int = 32
    tile_overlap_factor: float = 0.25
    scaling_factor: float = 0.493
    disc_off_grad_ckpt: bool = False
    time_compression_ratio: int=4
    spatial_compression_ratio: int=32

@dataclass
class AutoencoderVideoDCCfg:
    name: Literal["video_dc"]
    pretrained_from: str | None
    kwargs: VideoDCKwargsCfg
    