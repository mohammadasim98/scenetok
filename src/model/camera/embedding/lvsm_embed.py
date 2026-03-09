import os
import torch

from pathlib import Path
from torch import nn, Tensor
from dataclasses import dataclass
from jaxtyping import Float, Bool
from typing import Optional, Union, Literal
from dataclasses import dataclass
from einops import rearrange
from einops.layers.torch import Rearrange

from .embedding import Embedding

def zero_initialize(layer):
    if hasattr(layer, 'weight') and layer.weight is not None:
        nn.init.zeros_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        nn.init.zeros_(layer.bias)


@dataclass
class LVSMEmbeddingCfg:
    name: Literal["lvsm"]
    patch_size: int = 16
    in_channels: int = 6

class LVSMEmbedding(Embedding[LVSMEmbeddingCfg]):
    cfg: LVSMEmbeddingCfg
    model: nn.Module
    
    def __init__(
        self,
        cfg: LVSMEmbeddingCfg,
        embed_dim: int,
        input_shape: Union[tuple[int, int], list[int, int]],
        temporal_downsample: int=1

    ):
        super().__init__(cfg)
        self.input_shape = input_shape
        print("(LVSM Embedding) Temporal Downsample: ", temporal_downsample)
        self.model = nn.Sequential(
            Rearrange(
                "b ... (t c) (hh ph) (ww pw) -> b ... (hh ww) (t ph pw c)",
                ph=cfg.patch_size,
                pw=cfg.patch_size,
                t=temporal_downsample
            ),
            nn.Linear(
                cfg.in_channels * (cfg.patch_size**2),
                embed_dim,
                bias=False,
            ),
            # nn.RMSNorm(embed_dim),
        )

        # self.model.apply(zero_initialize)

    @property
    def output_shape(self) -> Union[tuple[int, int], list[int, int]]:
        return (
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

        # with torch.autocast(enabled=False, dtype=torch.float32, device_type="cuda"):
        x = self.model(x)
        x = rearrange(x, "b ... (h w) c -> b ... c h w", h=self.output_shape[0], w=self.output_shape[1])

        return x