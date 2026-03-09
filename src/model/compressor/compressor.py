import os
import torch

from pathlib import Path
from torch import nn, Tensor
from dataclasses import dataclass
from jaxtyping import Float
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from ..types import CompressorInputs

T = TypeVar("T")


class Compressor(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(
        self, 
        cfg: T
    ) -> None:
        
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def load_weights(
        self,
        path: Path | str,
        **kwargs
    ):  
        raise NotImplementedError()

    @abstractmethod
    def _forward(
        self,
        inputs: CompressorInputs, 
        latent_input: bool=False,
        return_qk: bool=False,
    ) -> Float[Tensor, "batch num dim"]:
        
        raise NotImplementedError()
    
    @torch.compile(disable=bool(int(os.getenv("DEBUG", 0))), fullgraph=True)
    def forward(
        self,
        inputs: CompressorInputs
    ) -> Float[Tensor, "batch num dim"]:
        
        return self._forward(
            inputs=inputs
        )
