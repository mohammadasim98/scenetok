import os
import torch

from pathlib import Path
from torch import nn, Tensor
from dataclasses import dataclass
from jaxtyping import Float
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from ..types import SceneGeneratorInputs



T = TypeVar("T")


class SceneGenerator(nn.Module, ABC, Generic[T]):
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
        inputs: SceneGeneratorInputs,
        **kwargs
    ) -> Float[Tensor, "batch view channel height width"]:
        
        raise NotImplementedError()
    
    @torch.compile(disable=bool(int(os.getenv("DEBUG", 0))), fullgraph=True)
    def forward(
        self,
        inputs: SceneGeneratorInputs,
        **kwargs
    ) -> Float[Tensor, "batch view channel height width"]:
        
        return self._forward(inputs, **kwargs)