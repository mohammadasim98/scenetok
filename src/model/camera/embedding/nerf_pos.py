import os
import math
import torch

from torch import Tensor
from dataclasses import dataclass
from jaxtyping import Float
from typing import Literal
from dataclasses import dataclass
from einops import rearrange


from .embedding import Embedding

@dataclass
class NeRFPositionalEmbeddingCfg:
    name: Literal["nerf"]
    freq_origin: int = 15
    freq_direction: int = 15,


class NeRFPositionalEmbedding(Embedding[NeRFPositionalEmbeddingCfg]):
    cfg: NeRFPositionalEmbeddingCfg
    
    def __init__(
        self, 
        cfg: NeRFPositionalEmbeddingCfg,
        **kwargs
    ) -> None:
        super().__init__(cfg)


    @property
    def output_dim(self) -> int:
        return 6 * (self.cfg.freq_origin + self.cfg.freq_direction)

    @property
    def input_dim(self) -> int:
        return 6

    def load_weights(
        self
    ):  
        pass

    def _nerf_pos_encoding(
        self, 
        x: Float[Tensor, "... dim1"], 
        freq: int
    ) -> Float[Tensor, "... dim2"]:
        scale = (
            2 ** torch.linspace(0, freq - 1, freq, device=x.device, dtype=x.dtype)
            * math.pi
        )
        x = x * scale
        return torch.sin(torch.cat([x, x + 0.5 * math.pi], dim=-1))
    
    def to_pos_encoding(
        self,
        pose: Float[Tensor, "... dim1"],
        freq_origin: int = 15,
        freq_direction: int = 15
    ) -> Float[Tensor, "... dim2"]:
        """
        Returns the rays represented as positional encoding. Follows NeRF to map the rays into a higher-dimensional space.
        Args:
            freq_origin (int): The frequency for the origin.
            freq_direction (int): The frequency for the direction.
            return_rays (bool): Whether to return the rays tensor or not.
        Returns:
            torch.Tensor: The rays tensor. Shape (B, T, H, W, 6 * (freq_origin + freq_direction)).
        """
        # pose = rearrange(pose, "b v c h w -> b v h w c")
        encoding = torch.cat(
            [
                self._nerf_pos_encoding(pose[..., :3], freq_origin),
                self._nerf_pos_encoding(pose[..., 3:], freq_direction),
            ],
            dim=-1,
        )
        # encoding = rearrange(encoding, "b v h w c -> b v c h w")
        return encoding
    
    def forward(
        self, 
        x: Float[Tensor, "... dim1 _ _"] 
    ) -> Float[Tensor, "... dim2 _ _"]:
        """
        Args:
            x: tensor to be patchified of shape (*B, C, H, W)
        Returns:
            patchified tensor of shape (*B, num_patches, embed_dim)
        """
        x = rearrange(x, "... c h w -> ... h w c")
        x = self.to_pos_encoding(x)
        x = rearrange(x, "... h w c -> ... c h w")

        return x

