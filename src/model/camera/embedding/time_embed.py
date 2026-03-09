import os
import math
import torch

from pathlib import Path
from torch import nn, Tensor
from dataclasses import dataclass
from jaxtyping import Float, Bool
from typing import Optional, Union, Literal
from dataclasses import dataclass, field
from einops import rearrange
from diffusers.models.embeddings import TimestepEmbedding

from .embedding import Embedding

@dataclass
class TimeEmbeddingCfg:
    name: Literal["time_embed"]
    in_channels: int=6
    dropout_prob: float=0.0
    fourier: bool=True
    freq_origin: int = 15
    freq_direction: int = 15
    timestep_embedding_kwargs: dict = field(default_factory=dict)

class RandomEmbeddingDropout(nn.Module):
    """
    Randomly nullify the input embeddings with a given probability.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, emb: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Randomly nullify the input embeddings with a probability p during training. For inference, the embeddings are nullified only if mask is provided.
        Args:
            emb: input embeddings of shape (B, ...)
            mask: mask tensor of shape (B, ). Only allowed during inference. If provided, embeddings for masked batches will be zeroed.
        """
        if mask is not None:
            assert not self.training, "embedding mask is only allowed during inference"
            assert mask.ndim == 1, "embedding mask should be of shape (B,)"

        if self.training and self.p > 0:
            mask = torch.rand(emb.shape[:1], device=emb.device) < self.p
        if mask is not None:
            mask = rearrange(mask, "... -> ..." + " 1" * (emb.ndim - 1))
            emb = torch.where(mask, torch.zeros_like(emb), emb)
        return emb

class TimeEmbedding(Embedding[TimeEmbeddingCfg]):
    cfg: TimeEmbeddingCfg
    model: nn.Module
    
    def __init__(
        self,
        cfg: TimeEmbeddingCfg,
        embed_dim: int,
        **kwargs
    ):
        super().__init__(cfg)
        self.dropout = RandomEmbeddingDropout(p=self.cfg.dropout_prob)
        in_channels = cfg.in_channels
        if cfg.fourier:
            in_channels = in_channels * (cfg.freq_direction + cfg.freq_origin)
        self.model = TimestepEmbedding(in_channels, embed_dim, **cfg.timestep_embedding_kwargs)

    def _nerf_pos_encoding(
        self, 
        x: Float[Tensor, "... dim1"], 
        freq: int
    ) -> Float[Tensor, "... dim2"]:
        scale = (
            2 ** torch.linspace(0, freq - 1, freq, device=x.device, dtype=x.dtype)
            * math.pi
        )
        x = rearrange(x[..., None] * scale, "... i s -> ... (i s)")
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

    def load_weights(
        self,
        path: Path | str,
        **kwargs
    ):  
        if self.model is None or path is None:
            return
        weights = torch.load(path)
        self.model.load_state_dict(weights, **kwargs)

    def forward(
        self, 
        x: Float[Tensor, "batch ... channel _ _"], 
        mask: Optional[Bool[Tensor, "batch"]] = None
    ) -> Float[Tensor, "batch ... channel _ _"]:
        """
        Args:
            x: tensor to be patchified of shape (*B, C, H, W)
        Returns:
            patchified tensor of shape (*B, num_patches, embed_dim)
        """
        *_, h, w = x.shape
        x = rearrange(x, "... c h w -> ... (h w) c ")
        if self.cfg.fourier:
            x = self.to_pos_encoding(x)
        x = self.model(x)
        x = self.dropout(x, mask)
        x = rearrange(x, "... (h w) c -> ... c h w", h=h, w=w)
        return x