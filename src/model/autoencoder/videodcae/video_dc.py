
import torch
from torch import Tensor
from pathlib import Path
from torch import nn
from safetensors.torch import load_file
from typing import Callable, Optional
from jaxtyping import Float
from huggingface_hub import PyTorchModelHubMixin
from einops import rearrange
from .models.dc_ae import DCAE, DCAEConfig, dc_ae_f32
from ..autoencoder_videodc import VideoDCKwargsCfg

__all__ = ["create_dc_ae_model_cfg", "DCAE_HF", "DC_AE"]


REGISTERED_DCAE_MODEL: dict[str, tuple[Callable, Optional[str]]] = {
    "dc-ae-f32t4c128": (dc_ae_f32, None),
}


def create_dc_ae_model_cfg(name: str, pretrained_path: Optional[str] = None) -> DCAEConfig:
    assert name in REGISTERED_DCAE_MODEL, f"{name} is not supported"
    dc_ae_cls, default_pt_path = REGISTERED_DCAE_MODEL[name]
    pretrained_path = default_pt_path if pretrained_path is None else pretrained_path
    model_cfg = dc_ae_cls(name, pretrained_path)
    return model_cfg


class DCAE_HF(DCAE, PyTorchModelHubMixin):
    def __init__(self, model_name: str):
        cfg = create_dc_ae_model_cfg(model_name)
        DCAE.__init__(self, cfg)


class AutoencoderVideoDC(nn.Module):
        
    
    def __init__(self, cfg: VideoDCKwargsCfg, img_size=256, horizon_flip=0.5, fp16=True):
        """Initialize VA_VAE
        Args:
            config: Configuration dict containing img_size, horizon_flip and fp16 parameters
        """
        super().__init__()
        
        self.model = DCAE_HF(
            cfg.model_name
        )
        self.model.scaling_factor = None
        self.model.use_temporal_tiling = cfg.use_temporal_tiling
        self.model.use_spatial_tiling = cfg.use_spatial_tiling
        self.model.spatial_tile_size = cfg.spatial_tile_size
        self.model.temporal_tile_size = cfg.temporal_tile_size
        self.model.tile_overlap_factor = cfg.tile_overlap_factor
        self.model.time_compression_ratio = cfg.time_compression_ratio
        self.model.spatial_compression_ratio = cfg.spatial_compression_ratio

        assert (
            cfg.spatial_tile_size // cfg.spatial_compression_ratio
        ), f"spatial tile size {cfg.spatial_tile_size} must be divisible by spatial compression of {cfg.spatial_compression_ratio}"
        assert (
            cfg.temporal_tile_size // cfg.time_compression_ratio
        ), f"temporal tile size {cfg.temporal_tile_size} must be divisible by temporal compression of {cfg.time_compression_ratio}"
        
        
        self.model.spatial_tile_latent_size = cfg.spatial_tile_size // cfg.spatial_compression_ratio
        self.model.temporal_tile_latent_size = cfg.temporal_tile_size // cfg.time_compression_ratio
        
    def from_pretrained(self, ckpt_path, **kwargs):
        
        weights = load_file(ckpt_path)
        self.model.load_state_dict(weights, strict=True)

        return self
    

    def encode(
        self, 
        x: Float[Tensor, "batch view channel height width"]
    ) -> Float[Tensor, "batch _ _ height width"]:
        
        b, v, c, h, w = x.shape
        x = rearrange(x, "b v c h w -> b c v h w")
        x = self.model.encode(x)
        x = rearrange(x, "b c v h w -> b v c h w")
        return x
    
    def decode(
        self, 
        x: Float[Tensor, "batch view channel height width"]
    ) -> Float[Tensor, "batch _ _ height width"]:
        
        b, v, c, h, w = x.shape
        x = rearrange(x, "b v c h w -> b c v h w")
        x = self.model.decode(x)
        x = rearrange(x, "b c v h w -> b v c h w")
        return x


