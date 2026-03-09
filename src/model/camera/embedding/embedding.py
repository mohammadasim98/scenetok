import os
import torch

from pathlib import Path
from torch import nn, Tensor
from jaxtyping import Float
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar



T = TypeVar("T")


class Embedding(nn.Module, ABC, Generic[T]):
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
    def forward(
        self
    ):
        raise NotImplementedError()