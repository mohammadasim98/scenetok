import os
import torch

from pathlib import Path
from torch import nn, Tensor
from dataclasses import dataclass
from jaxtyping import Float, Bool
from typing import Optional, Union, Literal
from dataclasses import dataclass, field
from einops import rearrange
from diffusers.models.embeddings import PatchEmbed

from .embedding import Embedding

@dataclass
class PatchEmbeddingCfg:
    name: Literal["patch_embed"]
    dropout_prob: float=0.0
    patch_size: int = 16
    in_channels: int = 6
    bias: bool = True
    flatten: bool = True
    patch_embed_kwargs: dict = field(default_factory=dict)

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

class PatchEmbedding(Embedding[PatchEmbeddingCfg]):
    cfg: PatchEmbeddingCfg
    model: nn.Module
    
    def __init__(
        self,
        cfg: PatchEmbeddingCfg,
        embed_dim: int,
        input_shape: Union[tuple[int, int], list[int, int]],
        **kwargs
    ):
        super().__init__(cfg)
        self.input_shape = input_shape
        self.dropout = RandomEmbeddingDropout(p=self.cfg.dropout_prob)
        self.model = PatchEmbed(
            height=input_shape[0],
            width=input_shape[1],
            patch_size=self.cfg.patch_size,
            in_channels=self.cfg.in_channels,
            embed_dim=embed_dim,
            bias=self.cfg.bias,
            flatten=self.cfg.flatten,
            pos_embed_max_size=max(input_shape), # In case input is not a square
            **self.cfg.patch_embed_kwargs,
        )
        self.ndim = 3 if self.cfg.flatten else 4
        self.output_shape = (
            self.input_shape[0] // self.cfg.patch_size,
            self.input_shape[1] // self.cfg.patch_size,
        )

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
        orig_shape = x.shape

        x = rearrange(x, "... c h w -> (...) c h w")
        x = self.model(x)
        x = x.reshape(*orig_shape[:-3], *x.shape[-self.ndim + 1 :])
        x = self.dropout(x, mask)
        x = rearrange(x, "... (h w) c -> ... c h w", h=self.output_shape[0], w=self.output_shape[1])
        return x