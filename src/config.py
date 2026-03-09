from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .dataset.data_module import DataLoaderCfg, DatasetCfg
from .model.config import FreezeCfg, OptimizerCfg, TestCfg, TrainCfg, ModelCfg, ValCfg
from .profiler import ProfilerCfg
from .model.sampler import SamplerCfg
@dataclass
class CheckpointingCfg:
    load: Optional[str]  # Not a path, since it could be something like wandb://...
    every_n_train_steps: int
    save_top_k: int
    resume: bool = False
    save: bool = True


@dataclass
class TrainerCfg:
    max_steps: int
    val_check_interval: int | float | None
    gradient_clip_val: int | float | None
    task_steps: int | None
    precision: Literal[16, 32, 64] | Literal["16-true", "16-mixed", "bf16-true", "bf16-mixed", "32-true", "64-true"] | Literal["bf16", "16", "32", "64"] | None = None
    validate: bool = True
    accumulate_grad_batches: int=1
    limit_test_batches: int | None=32
    strategy: str = 'ddp_find_unused_parameters_true'
    devices: int=1
    num_nodes: int=1
    profiler: ProfilerCfg = None
    
@dataclass
class RootCfg:
    wandb: dict
    mode: Literal["train", "val", "test", "predict_train", "predict_test"]
    dataset: DatasetCfg
    model: ModelCfg
    data_loader: DataLoaderCfg
    optimizer: OptimizerCfg
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    test: TestCfg
    train: TrainCfg
    val: ValCfg
    freeze: FreezeCfg
    seed: int | None
    sampler: SamplerCfg | None = None

TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )

def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg,
        {},
    )
