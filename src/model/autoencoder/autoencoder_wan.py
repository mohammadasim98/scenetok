from typing import Literal, Optional, Tuple, List
from dataclasses import dataclass
from typing import Union
import torch
from src.model.autoencoder.wanvae import Wan2_2_VAE
from typing import Callable, Optional
from jaxtyping import Float
from torch import Tensor, nn
from einops import rearrange
# Refer to https://github.com/huggingface/diffusers/blob/6131a93b969f87d171148bd367fd9990d5a49b6b/src/diffusers/models/autoencoders/autoencoder_kl.py#L38
@dataclass
class WanKwargsCfg:
    
    in_channels: int = 3
    latent_channels: int = 48
    scaling_factor: float = 1.0

@dataclass
class AutoencoderWanCfg:
    name: Literal["wan", "wan_single"]
    pretrained_from: str | None
    kwargs: WanKwargsCfg
    


class AutoencoderWan(nn.Module):
        
    
    def __init__(self, cfg: WanKwargsCfg):
        """Initialize VA_VAE
        Args:
            config: Configuration dict containing img_size, horizon_flip and fp16 parameters
        """
        super().__init__()
        self.model = Wan2_2_VAE(
            vae_pth=None
        )

        

    def from_pretrained(self, ckpt_path, **kwargs):
        print("(Wan) Loading from pretrained weights from: ", ckpt_path)
        self.model.model.load_state_dict(
            torch.load(ckpt_path, map_location=torch.device("cpu")), assign=True)

        return self
    

    def encode(
        self, 
        x: Float[Tensor, "batch view channel height width"]
    ) -> Float[Tensor, "batch _ _ height width"]:
        
        b, v, c, h, w = x.shape
        scale = [self.model.scale[0].to(x.device), self.model.scale[1].to(x.device)]
        x = rearrange(x, "b v c h w -> b c v h w")
        x = self.model.model.encode(x, scale)
        x = rearrange(x, "b c v h w -> b v c h w")
        return x
    
    def decode(
        self, 
        x: Float[Tensor, "batch view channel height width"]
    ) -> Float[Tensor, "batch _ _ height width"]:
        
        b, v, c, h, w = x.shape
        scale = [self.model.scale[0].to(x.device), self.model.scale[1].to(x.device)]
        x = rearrange(x, "b v c h w -> b c v h w")
        x = self.model.model.decode(x, scale)
        x = rearrange(x, "b c v h w -> b v c h w")
        return x