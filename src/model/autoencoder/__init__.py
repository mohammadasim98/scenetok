import torch
from dataclasses import dataclass
from typing import Union, Literal, List, Tuple, Dict, Optional
from diffusers import AutoencoderKL, AutoencoderDC, AutoencoderKLLTXVideo
from dataclasses import asdict

from .autoencoder_kl import AutoencoderKLCfg
from .autoencoder_kl_ltxvideo import AutoencoderKLLTXVideoCfg
from .autoencoder_dc import AutoencoderDCCfg
from .autoencoder_va import AutoencoderVACfg
from .autoencoder_videodc import AutoencoderVideoDCCfg
from .autoencoder_wan import AutoencoderWanCfg, AutoencoderWan
from .vavae import AutoencoderVA
from .videodcae import AutoencoderVideoDC

AutoencoderCfg =  AutoencoderKLCfg | AutoencoderDCCfg | AutoencoderVACfg | AutoencoderVideoDCCfg | AutoencoderKLLTXVideoCfg | AutoencoderWanCfg
Autoencoder = AutoencoderKL | AutoencoderDC | AutoencoderVA | AutoencoderVideoDC | AutoencoderKLLTXVideo | AutoencoderWan

@dataclass
class AutoencodersCfg:
    target: Optional[AutoencoderCfg] = None
    context: Optional[AutoencoderCfg] = None


AUTOENCODERS = {
    "kl": AutoencoderKL,
    "dc": AutoencoderDC,
    "va": AutoencoderVA,
    "video_dc": AutoencoderVideoDC,
    "ltx": AutoencoderKLLTXVideo,
    "wan": AutoencoderWan,
    "wan_single": AutoencoderWan
}


def _parse_cfg(cfg: AutoencoderCfg):

    # Original type and types returned by hydra may conflict
    if type(cfg.down_block_types) == list:
        cfg.down_block_types = tuple(cfg.down_block_types)
    if type(cfg.up_block_types) == list:
        cfg.up_block_types = tuple(cfg.up_block_types)
    if type(cfg.block_out_channels) == list:
        cfg.block_out_channels = tuple(cfg.block_out_channels)
    if cfg.latents_mean is not None and type(cfg.latents_mean) == list:
        cfg.latents_mean = tuple(cfg.latents_mean)
    if cfg.latents_std is not None and type(cfg.latents_std) == list:
        cfg.latents_std = tuple(cfg.latents_std)
    
    return cfg

def get_autoencoder(
    cfg: AutoencoderCfg,
    **kwargs
) -> Autoencoder:
    if cfg.name in ["va", "video_dc", "wan", "wan_single"]:
        
        print(f"(Autoencoder) ({cfg.name}) Loading pretrained weights from: ", cfg.pretrained_from)
        return AUTOENCODERS[cfg.name](cfg.kwargs).from_pretrained(cfg.pretrained_from)
    else:
        if cfg.pretrained_from is None:
            return AUTOENCODERS[cfg.name](**asdict(_parse_cfg(cfg.kwargs)))
        else:
            print(f"(Autoencoder) ({cfg.name}) Loading pretrained weights from: ", cfg.pretrained_from)
            return AUTOENCODERS[cfg.name].from_pretrained(cfg.pretrained_from, subfolder=cfg.subfolder)

