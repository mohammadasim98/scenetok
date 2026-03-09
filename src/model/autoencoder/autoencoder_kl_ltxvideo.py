from typing import Literal, Optional, Tuple, List
from dataclasses import dataclass
from typing import Union
# Refer to https://github.com/huggingface/diffusers/blob/6131a93b969f87d171148bd367fd9990d5a49b6b/src/diffusers/models/autoencoders/autoencoder_kl.py#L38
@dataclass
class KLLTXVideoKwargsCfg:
    
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 128
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
    down_block_types: Tuple[str, ...] = (
        "LTXVideoDownBlock3D",
        "LTXVideoDownBlock3D",
        "LTXVideoDownBlock3D",
        "LTXVideoDownBlock3D",
    )
    decoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: Tuple[int, ...] = (4, 3, 3, 3, 4)
    decoder_layers_per_block: Tuple[int, ...] = (4, 3, 3, 3, 4)
    spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False)
    decoder_spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False)
    decoder_inject_noise: Tuple[bool, ...] = (False, False, False, False, False)
    downsample_type: Tuple[str, ...] = ("conv", "conv", "conv", "conv")
    upsample_residual: Tuple[bool, ...] = (False, False, False, False)
    upsample_factor: Tuple[int, ...] = (1, 1, 1, 1)
    timestep_conditioning: bool = False
    patch_size: int = 4
    patch_size_t: int = 1
    resnet_norm_eps: float = 1e-6
    scaling_factor: float = 1.0
    encoder_causal: bool = True
    decoder_causal: bool = False
    spatial_compression_ratio: int = None
    temporal_compression_ratio: int = None

@dataclass
class AutoencoderKLLTXVideoCfg:
    name: Literal["kl_ltx_video"]
    subfolder: str | None
    pretrained_from: str | None
    kwargs: KLLTXVideoKwargsCfg
    