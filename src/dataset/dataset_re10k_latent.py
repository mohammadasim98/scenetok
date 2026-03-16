import os
import re
import json
import torch
import numpy as np
from pathlib import Path
from typing import Literal
import torchvision.transforms as tf
from dataclasses import asdict, dataclass
from functools import cached_property
from torch.utils.data import IterableDataset

from .dataset import DatasetCfgCommon
from .dtypes import Stage
from .view_sampler import ViewSampler
from src.misc.camera_utils import fps_from_pose


@dataclass
class DatasetRE10kHybridTemporalCfg(DatasetCfgCommon):
    name: Literal["re10k_precomp"]
    target_root: Path | None
    context_root: Path | None
    baseline_epsilon: float
    max_fov: float
    make_baseline: bool
    limit_frame_distance: list | None = None

def load_latents(npz_path: str):
    data = np.load(npz_path)
    latents = torch.from_numpy(data['latent']).float()  # Convert to float32 for use
    extrinsics = torch.from_numpy(data['extrinsics']).float()
    intrinsics = torch.from_numpy(data['intrinsics']).float()
    metadata = {
        "near": torch.from_numpy(data['near']).float(),
        "far": torch.from_numpy(data['far']).float(),
        "index": data['index'].tolist(),
        "flipped": bool(data['flipped']),
    }

    return latents, extrinsics, intrinsics, metadata

def load_images(npz_path: str):
    data = np.load(npz_path)
    images = torch.from_numpy(data['image']).float()  # Convert to float32 for use
    extrinsics = torch.from_numpy(data['extrinsics']).float()
    intrinsics = torch.from_numpy(data['intrinsics']).float()
    metadata = {
        "near": torch.from_numpy(data['near']).float(),
        "far": torch.from_numpy(data['far']).float(),
        "index": data['index'].tolist(),
        "flipped": bool(data['flipped']),
    }

    return images / 255, extrinsics, intrinsics, metadata

def reflect_image(image):
    return image.flip(-1)

class DatasetRE10kHybridTemporal(IterableDataset):
    cfg: DatasetRE10kHybridTemporalCfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor
    chunks: list[Path]


    def __init__(
        self,
        cfg: DatasetRE10kHybridTemporalCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        force_shuffle: bool = False
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self.force_shuffle = force_shuffle

        self.context_chunks = []
        self.target_chunks = []

        self.context_root = cfg.context_root / self.data_stage
        self.target_root = cfg.target_root / self.data_stage
        
        if cfg.limit_frame_distance is not None:
            self.chunks = []
            for dist in cfg.limit_frame_distance:
                chunks = sorted(
                    [path.name for path in self.target_root.iterdir() if path.suffix == ".npz" and f"_{dist}_" in path.name]
                )
                self.chunks.extend(chunks)
        else: 
            self.chunks = sorted(
                [path.name for path in self.target_root.iterdir() if path.suffix == ".npz"]
            )

        print(len(self.chunks), "chunks found for", self.data_stage)
    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in ("train", "val") or self.force_shuffle:
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:

            # split dir + filename
            dirname, fname = os.path.split(chunk_path)
            # remove the decimal/number before "_flipped"
            new_fname = re.sub(r"_[0-9]+_[0-9]+(?=(_flipped)?\.h5.npz$)", "", fname)
            # new_fname = fname
            # if "flipped" in fname:
            #     new_fname = new_fname.replace("_flipped", "")

            new_chunk_path = os.path.join(dirname, new_fname)
            new_chunk_path = new_chunk_path.replace(".h5", "")
            
            try:
                context_latents, context_extrinsics, context_intrinsics, context_metadata = load_latents(self.context_root / new_chunk_path)
            except FileNotFoundError as e:
                print(f"Context file not found: {self.context_root / new_chunk_path}")
                continue
                
            target_latents, target_extrinsics, target_intrinsics, target_metadata = load_latents(self.target_root / chunk_path)

            scene = chunk_path.split(".")[0]
            num_views = target_extrinsics.shape[0]
            num_latents = target_latents.shape[0]
            total_views = context_latents.shape[0]

            try:
                view_indices, upsampled_indices = self.view_sampler.sample(num_views, num_latents, stage=self.stage, extrinsics=context_extrinsics)
            except ValueError as err:

                continue
            
            sample = {"scene": scene}

            context_indices = fps_from_pose(context_extrinsics, self.cfg.view_sampler.num_context_views)
            if context_indices is not None:

                sample["context"] = {
                    "extrinsics": context_extrinsics[context_indices],
                    "intrinsics": context_intrinsics[context_indices],
                    "latent": context_latents[context_indices],
                    "index": context_indices
                }

            if view_indices.target is not None:
                sample["target"] = {
                    "extrinsics": target_extrinsics[upsampled_indices],
                    "intrinsics": target_intrinsics[upsampled_indices],
                    "latent": target_latents[view_indices.target],
                    "index": upsampled_indices
                }
                
            yield sample

    
    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

