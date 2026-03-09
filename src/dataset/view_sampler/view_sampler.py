from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, Literal

import torch
from jaxtyping import Int64
from torch import Generator, Tensor

from ...misc.step_tracker import StepTracker
from ..dtypes import Stage

T = TypeVar("T")


@dataclass
class ViewIndex:
    context: Int64[Tensor, " context_view"]             # indices for context views
    target: Int64[Tensor, " target_view"] | None = None # indices for target views

@dataclass
class ViewSamplerCfg:
    num_context_views: int
    num_target_views: int
    context_sampling: Literal["farthest_point", "uniform"]
    

class ViewSampler(ABC, Generic[T]):
    cfg: T
    stage: Stage
    cameras_are_circular: bool
    step_tracker: StepTracker | None
    generator: Generator | None

    def __init__(
        self,
        cfg: T,
        stage: Stage,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
        generator: Generator | None = None
    ) -> None:
        self.cfg = cfg
        self.stage = stage
        self.cameras_are_circular = cameras_are_circular
        self.step_tracker = step_tracker
        self.generator = generator

    @abstractmethod
    def sample(
        self,
        scene: str,
        num_views: int,
        device: torch.device = torch.device("cpu"),
        **kwargs
    ) -> list[ViewIndex]:
        pass

    @property
    @abstractmethod
    def num_target_views(self) -> int:
        pass

    @property
    @abstractmethod
    def num_context_views(self) -> int:
        pass

    @property
    def global_step(self) -> int:
        return 0 if self.step_tracker is None else self.step_tracker.get_step()
