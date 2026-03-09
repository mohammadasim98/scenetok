
from pathlib import Path
from torch import nn
from abc import ABC, abstractmethod
from typing import Generic, TypeVar


from ..types import CameraInputs

T = TypeVar("T")


class Camera(nn.Module, ABC, Generic[T]):
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
        self: CameraInputs,
        **kwargs
    ):
        raise NotImplementedError()
