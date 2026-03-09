import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Generator
from dacite import Config, from_dict

from ...evaluation.types import IndexEntry
from ...misc.step_tracker import StepTracker
from ..dtypes import Stage
from .view_sampler import ViewIndex, ViewSampler


@dataclass
class ViewSamplerEvaluationVideoCfg:
    name: Literal["evaluation_video"]
    index_path: Path
    num_context_views: int
    num_target_views: int
    temporal_downsample: int=4


class ViewSamplerEvaluationVideo(ViewSampler[ViewSamplerEvaluationVideoCfg]):
    index: dict[str, list[IndexEntry]]

    def __init__(
        self,
        cfg: ViewSamplerEvaluationVideoCfg,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
        generator: Generator | None = None
    ) -> None:
        super().__init__(cfg, stage, cameras_are_circular, step_tracker, generator)

        dacite_config = Config(cast=[tuple])
        with cfg.index_path.open("r") as f:
            self.index = {
                k: [from_dict(IndexEntry, v, dacite_config)]
                for k, v in json.load(f).items()
            }
            self.total_samples = sum(len(views) for views in self.index.values())
    
    def sample(
        self,
        num_views: int,
        num_latents: int,
        scene: str,
        device: torch.device = torch.device("cpu"),
        **kwargs
    ) -> list[ViewIndex]:
        entries = self.index.get(scene)
        
        if not entries:
            raise ValueError(f"No indices available for scene {scene}.")
        entry = entries[0]
        context_indices = torch.tensor(entry.context, dtype=torch.int64, device=device)
        target_indices = torch.tensor(entry.target, dtype=torch.int64, device=device)
        latent_indices = target_indices // self.cfg.temporal_downsample

        return ViewIndex(
            context_indices,
            target_indices
        ), latent_indices

    @property
    def num_context_views(self) -> int:
        return 0

    @property
    def num_target_views(self) -> int:
        return 0
