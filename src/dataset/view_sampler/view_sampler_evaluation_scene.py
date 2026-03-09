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
class ViewSamplerEvaluationSceneCfg:
    name: Literal["evaluation_scene"]
    test_index_path: Path
    train_index_path: Path
    num_context_views: int=12
    num_target_views: int=8

class ViewSamplerEvaluationScene(ViewSampler[ViewSamplerEvaluationSceneCfg]):
    index: dict[str, list[IndexEntry]]

    def __init__(
        self,
        cfg: ViewSamplerEvaluationSceneCfg,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
        generator: Generator | None = None
    ) -> None:
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular, step_tracker, generator)

        dacite_config = Config(cast=[tuple])
        self.index = {"test": {}, "train": {}}
        with cfg.test_index_path.open("r") as f:
            self.index["test"] = {
                k: views
                for k, views in json.load(f).items()
            }
            self.total_samples = sum(len(views) for views in self.index.values())

        with cfg.train_index_path.open("r") as f:
            self.index["train"] = {
                k: views
                for k, views in json.load(f).items()
            }
            self.total_samples = sum(len(views) for views in self.index.values())


    def sample(
        self,
        scene: str,
        stage: str,
        device: torch.device = torch.device("cpu"),
    ) -> list[ViewIndex]:
        entries = self.index[stage].get(scene)
        if not entries:
            raise ValueError(f"No indices available for scene {scene}.")
        return [
            ViewIndex(
                torch.tensor(entry.context, dtype=torch.int64, device=device),
                torch.tensor(entry.target, dtype=torch.int64, device=device)
            )
            for entry in entries
        ]

    @property
    def num_context_views(self) -> int:
        return 0

    @property
    def num_target_views(self) -> int:
        return 0
