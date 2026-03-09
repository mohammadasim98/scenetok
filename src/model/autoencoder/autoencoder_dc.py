from typing import Literal, Optional, Tuple, List
from dataclasses import dataclass
from typing import Union
# Refer to https://github.com/huggingface/diffusers/blob/6131a93b969f87d171148bd367fd9990d5a49b6b/src/diffusers/models/autoencoders/autoencoder_kl.py#L38
@dataclass
class DCKwargsCfg:
    
    in_channels: int = 3
    latent_channels: int = 32
    attention_head_dim: int = 32
    encoder_block_types: Union[str, Tuple[str], List[str]] = "ResBlock"
    decoder_block_types: Union[str, Tuple[str], List[str]] = "ResBlock"
    encoder_block_out_channels: Union[Tuple[int], List[int]] = (128, 256, 512, 512, 1024, 1024)
    decoder_block_out_channels: Union[Tuple[int], List[int]] = (128, 256, 512, 512, 1024, 1024)
    encoder_layers_per_block: Union[Tuple[int], List[int]] = (2, 2, 2, 3, 3, 3)
    decoder_layers_per_block: Union[Tuple[int], List[int]] = (3, 3, 3, 3, 3, 3)
    encoder_qkv_multiscales: Union[Tuple[Tuple[int, ...], ...], List[List[int]]] = ((), (), (), (5,), (5,), (5,))
    decoder_qkv_multiscales: Union[Tuple[Tuple[int, ...], ...], List[List[int]]] = ((), (), (), (5,), (5,), (5,))
    upsample_block_type: str = "pixel_shuffle"
    downsample_block_type: str = "pixel_unshuffle"
    decoder_norm_types: Union[str, Tuple[str], List[str]] = "rms_norm"
    decoder_act_fns: Union[str, Tuple[str], List[str]] = "silu"
    scaling_factor: float = 1.0

@dataclass
class AutoencoderDCCfg:
    name: Literal["dc"]
    subfolder: str | None
    pretrained_from: str | None
    kwargs: DCKwargsCfg
    