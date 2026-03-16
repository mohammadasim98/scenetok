import os
import json
import torch
import numpy as np
import torchvision.transforms as tf

from pathlib import Path
from typing import Literal
from dataclasses import dataclass
from torch.utils.data import Dataset

# Custom modules
from .dtypes import Stage
from .dataset import DatasetCfgCommon
from .view_sampler import ViewSampler
from src.misc.data_utils import load_data
from src.misc.distributed_utils import _global_worker_info

@dataclass
class DatasetLatentCfg(DatasetCfgCommon):
    name: Literal["latent"]
    target_root: Path | None
    context_root: Path | None
    baseline_epsilon: float
    make_baseline: bool
    map_dict: Path | None
    limit_target_downsample_factor: list | None = None
    limit_context_downsample_factor: list | None = None
    random_flip: bool=True
    context_latent_type: Literal["va", "wan_single"] = "va"
    target_latent_type: Literal["videodc", "wan"] = "videodc"

class DatasetLatent(Dataset):
    cfg: DatasetLatentCfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor
    chunks: list[Path]

    def __init__(
        self,
        cfg: DatasetLatentCfg,
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

        self.map_dict = json.load(open(self.cfg.map_dict, "r"))
        self.scenes = list(self.map_dict[self.data_stage]["1"].keys())

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __getitem__(self, index):


        # NOTE: Chunk the data files across multiple workers and ranks to avoid conflicts and deadlocks
        N = len(self.scenes)
        gid, gcount = _global_worker_info()

        chunk = (N + gcount - 1) // gcount  # ceil
        start = gid * chunk
        end = min(start + chunk, N)

        # If N < gcount, some shards are empty; fallback safely
        if start >= end:
            start, end = 0, N

        idx = torch.randint(start, end, (1,)).item()

        # Make sure to set to correct index during test (non-iterable style)
        if self.stage == "test":
            idx = index

        # NOTE: Get context and target latents
        # Fetch a specific  scene
        key = self.scenes[idx]
        flip = 0
        if self.cfg.random_flip and self.data_stage == "train":
            flip = np.random.choice([False, True]).astype(int)

        downsample_idx = torch.randint(1, len(self.cfg.limit_context_downsample_factor)+1, (1, ))
        chunk = self.map_dict[self.data_stage][str(downsample_idx.item())][key][flip]
        
        # Get context chunk (different since we use video latents for targets)
        context_chunk = chunk
        context_downsample = str(self.cfg.limit_context_downsample_factor[downsample_idx-1])
        try:
            if os.path.exists(self.context_root / context_downsample / context_chunk):
                context_latents, context_extrinsics, context_intrinsics = load_data(self.context_root / context_downsample / context_chunk, key="latent")
            else:
                print(f"Refetching example due to file not found at {self.context_root}/{context_downsample}/{context_chunk}")
                return self.__getitem__(index)
        except:
            print(f"Refetching example due to an unkown error")

            return self.__getitem__(index)

        # Get target chunk
        downsample_idx = torch.randint(1, len(self.cfg.limit_target_downsample_factor)+1, (1, ))
        chunk = self.map_dict[self.data_stage][str(downsample_idx.item())][key][flip]

        target_chunk = chunk
        target_downsample = str(self.cfg.limit_target_downsample_factor[downsample_idx-1])
        try:
            if os.path.exists(self.target_root / target_downsample / target_chunk):
                target_latents, target_extrinsics, target_intrinsics = load_data(self.target_root / target_downsample / target_chunk, key="latent")
            else:
                print(f"Refetching example due to file not found at {self.target_root}/{target_downsample}/{target_chunk}")
                return self.__getitem__(index)
        except:
            print(f"Refetching example due to an unkown error")
            return self.__getitem__(index)
        

        # NOTE: Sample indices
        # Chunk name is the scene name
        scene = chunk
        # Get total number of views (different from number of latents due to temporal compression)
        num_views = target_extrinsics.shape[0]
        # Get total number of latents
        num_latents = target_latents.shape[0]

        # Sanity check to make sure number of views are the same as the number of latents (should not occur if latents are precomputed correctly)
        # If there is a discrepency, skip the sample and fetch another one
        if self.cfg.target_latent_type == "wan" and num_latents != ( 5 * (num_views // 17)):
            print(f"Number of views {num_views} is not the same as number of latents {num_latents}")
            return self.__getitem__(index)
        # if self.cfg.target_latent_type == "videodc" and num_latents != ( num_views // 16):
        #     print(f"Number of views {num_views} is not the same as number of latents {num_latents}")
        #     return self.__getitem__(index)
        
        # Sample context and target views and return the upsampled indices corresponding to actual frames (not latent frames)
        # This is need for indexing camera poses
        # view_indices contains latent indices for targets
        try:
            view_indices, latent_indices = self.view_sampler.sample(
                num_views=num_views, 
                num_latents=num_latents, 
                stage=self.stage, 
                extrinsics=context_extrinsics, 
                scene=scene
            )
        except ValueError as err:

            print(err)
            return self.__getitem__(index)
 
        # NOTE: Rescale scene if enabled and define the return datatype
        sample = {"scene": scene}

        # Scale the extrinsics positions based on the endpoints of the context views
        if self.cfg.make_baseline:
            ctxt_extrinsics = context_extrinsics[view_indices.context]
            a = ctxt_extrinsics[0, :3, 3]
            b = ctxt_extrinsics[-1, :3, 3]
            scale = (a - b).norm()
            if scale < self.cfg.baseline_epsilon:
                print(
                    f"Skipped {scene} because of insufficient baseline "
                    f"{scale:.6f}"
                )
                return
            target_extrinsics[:, :3, 3] /= scale
            context_extrinsics[:, :3, 3] /= scale
        
        # Index and return data
        if view_indices.context is not None:

            sample["context"] = {
                "extrinsics": context_extrinsics[view_indices.context].float(),
                "intrinsics": context_intrinsics[view_indices.context].float(),
                "latent": context_latents[view_indices.context].float(),
                "index": view_indices.context
            }

        if view_indices.target is not None:
            sample["target"] = {
                "extrinsics": target_extrinsics[view_indices.target].float(),
                "intrinsics": target_intrinsics[view_indices.target].float(),
                "latent": target_latents[latent_indices].float(),
                "index": view_indices.target
            }

        return sample

    
    @property
    def data_stage(self) -> Stage:

        if self.stage == "val":
            return "test"
        return self.stage

    def __len__(self) -> int:

        return len(self.scenes)
