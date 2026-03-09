import random
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import torch
from lightning import LightningDataModule
from torch import Generator, nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from ..misc.step_tracker import StepTracker
from . import DatasetCfg, get_dataset
from .dtypes import DataShim, Stage
from .validation_wrapper import ValidationWrapper
def safe_collate(batch):
    try:
        return torch.utils.data.default_collate(batch)
    except RuntimeError as e:
        if "resize storage that is not resizable" in str(e):
            print("💥 Collate failed")
            for i, b in enumerate(batch):
                if torch.is_tensor(b):
                    print(i, b.shape, b.device, b.dtype, b.is_contiguous())
                elif type(b) == dict:
                    
                    print(b["scene"])
                    print(b["context"].keys())
                    print(b["context"]["latent"].shape)
                    print(b["context"]["extrinsics"].shape)
                    print(b["context"]["intrinsics"].shape)
                    print(b["context"]["index"].shape)
                    print(b["target"].keys())
                    print(b["target"]["latent"].shape)
                    print(b["target"]["extrinsics"].shape)
                    print(b["target"]["intrinsics"].shape)
                    print(b["target"]["index"].shape)
                else:
                    print(b)
            raise
        else:
            raise

def get_data_shim(encoder: nn.Module) -> DataShim:
    """Get functions that modify the batch. It's sometimes necessary to modify batches
    outside the data loader because GPU computations are required to modify the batch or
    because the modification depends on something outside the data loader.
    """

    shims: list[DataShim] = []
    if hasattr(encoder, "get_data_shim"):
        shims.append(encoder.get_data_shim())

    def combined_shim(batch):
        for shim in shims:
            batch = shim(batch)
        return batch

    return combined_shim


@dataclass
class DataLoaderStageCfg:
    name: Literal["train", "test", "val"]
    shuffle: bool=True
    batch_size: int=16
    max_batches: int=-1
    num_workers: int=4
    prefetch_factor: int | None=None
    pin_memory: bool=False
    seed: int | None=None
    interval: int | None=None  
    persistent_workers: bool=False
    full_reference_metrics: list[str] | None = None
    general_reference_metrics: list[str] | None = None
@dataclass
class DataLoaderCfg:
    train: DataLoaderStageCfg
    test: DataLoaderStageCfg
    val: dict[str, DataLoaderStageCfg] = field(default_factory= lambda: {
        "standard": DataLoaderStageCfg(name="val", interval=5000, max_batches=1, batch_size=8),
        "evaluate": DataLoaderStageCfg(name="val", interval=50000, max_batches=32, batch_size=32, shuffle=False, seed=0)
    })


DatasetShim = Callable[[Dataset, Stage], Dataset]


def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class DataModule(LightningDataModule):
    dataset_cfg: DatasetCfg
    data_loader_cfg: DataLoaderCfg
    step_tracker: StepTracker | None
    dataset_shim: DatasetShim

    def __init__(
        self,
        dataset_cfg: DatasetCfg,
        data_loader_cfg: DataLoaderCfg,
        step_tracker: StepTracker | None = None,
        dataset_shim: DatasetShim = lambda dataset, _: dataset,
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.data_loader_cfg = data_loader_cfg
        self.step_tracker = step_tracker
        self.dataset_shim = dataset_shim
    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> bool | None:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def get_generator(self, loader_cfg: DataLoaderStageCfg) -> torch.Generator | None:
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed)
        return generator

    def train_dataloader(self):
        generator = self.get_generator(self.data_loader_cfg.train)
        dataset = get_dataset(self.dataset_cfg, "train", self.step_tracker, generator, force_shuffle=True)
        dataset = self.dataset_shim(dataset, "train")
        return DataLoader(
            dataset,
            self.data_loader_cfg.train.batch_size,
            shuffle=not isinstance(dataset, IterableDataset),
            num_workers=self.data_loader_cfg.train.num_workers,
            generator=generator,
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.train),
            prefetch_factor=self.data_loader_cfg.train.prefetch_factor,
            pin_memory=self.data_loader_cfg.train.pin_memory,
            drop_last=True,
            collate_fn=safe_collate
            
        )

    def val_dataloader(self) -> dict[str, DataLoader]:
        
        dataloaders = {}
        for i, (key, cfg) in enumerate(self.data_loader_cfg.val.items()):
            generator = self.get_generator(cfg)
            dataset = get_dataset(self.dataset_cfg, "val", self.step_tracker, generator, force_shuffle=True)
            dataset = self.dataset_shim(dataset, "val")
            dataloaders[key] = DataLoader(
                ValidationWrapper(dataset, cfg.batch_size) if i == 0 else dataset,
                cfg.batch_size,
                num_workers=cfg.num_workers,
                generator=generator,
                worker_init_fn=worker_init_fn,
                persistent_workers=self.get_persistent(cfg),
                prefetch_factor=cfg.prefetch_factor,
                pin_memory=cfg.pin_memory,
                shuffle=cfg.shuffle
            )
        return dataloaders

    def test_dataloader(self):
        generator = self.get_generator(self.data_loader_cfg.test)
        dataset = get_dataset(self.dataset_cfg, "test", self.step_tracker, generator, force_shuffle=True)
        dataset = self.dataset_shim(dataset, "test")
        return DataLoader(
            dataset,
            self.data_loader_cfg.test.batch_size,
            num_workers=self.data_loader_cfg.test.num_workers,
            generator=generator,
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.test),
            prefetch_factor=self.data_loader_cfg.test.prefetch_factor,
            pin_memory=self.data_loader_cfg.test.pin_memory
        )
        
    def predict_dataloader(self):
        generator = self.get_generator(self.data_loader_cfg.test)
        dataset = get_dataset(self.dataset_cfg, "test", self.step_tracker, generator, force_shuffle=False)
        dataset = self.dataset_shim(dataset, "test")
        return DataLoader(
            dataset,
            self.data_loader_cfg.test.batch_size,
            num_workers=self.data_loader_cfg.test.num_workers,
            generator=generator,
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.test)
        )
