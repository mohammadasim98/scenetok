from dataclasses import dataclass
from typing import Literal

import torch
import math
import numpy as np

from src.misc.camera_utils import fps_from_pose
from .view_sampler import ViewIndex, ViewSampler, ViewSamplerCfg

@dataclass
class ViewSamplerUnboundedCfg(ViewSamplerCfg):
    name: Literal["unbounded"]
    
    temporal_downsample: int=4
    temporal_tile_size: int=16
    chunk_index_gap: int=4
    offset: int=0
    min_context_views: int=1
    num_target_split: int=1
    target_split_prob: float=0.5
    sample_cond_views: bool=False
    max_cond_number: int=1
class ViewSamplerUnbounded(ViewSampler[ViewSamplerUnboundedCfg]):
    
    def schedule(
        self, 
        initial: int, 
        final: int,
        steps: int
    ) -> int:
        fraction = self.global_step / steps
        return min(initial + int((final - initial) * fraction), final)

    def sample(
        self,
        num_views: int,
        num_latents: int,
        stage: str,
        extrinsics: torch.Tensor,
        **kwargs
    ) -> tuple[ViewIndex, torch.Tensor]:
        nsamples = max(self.num_context_views, self.num_target_views * self.cfg.temporal_downsample)
        if num_latents < self.num_target_views:
            raise ValueError(f"Example has less number of frames --> {num_views} < {nsamples} and {num_latents} < {self.num_target_views}!")
        
        if stage == "train":
            chunk_index_gap = self.cfg.chunk_index_gap
        else:
            chunk_index_gap = 8


        num_target_views = self.num_target_views
        num_context_views = self.num_context_views
        num_target_split = self.cfg.num_target_split if stage == "train" else 1
        index_target = torch.arange(0, num_latents).long()

        starting_indices = torch.arange(0, num_latents - num_target_views + 1, chunk_index_gap)
        
        
        num_target_split = min(len(starting_indices), num_target_split)

        if np.random.choice([True, False], size=1, p=[1 - self.cfg.target_split_prob, self.cfg.target_split_prob]):
            num_target_split = 1
        
        idxs = torch.multinomial(torch.ones_like(starting_indices).float(), num_target_split, replacement=False)
        index_targets = []
        index_unrolled = []
        for idx in idxs:
            starting_index = starting_indices[idx]
            target = index_target[starting_index:starting_index + num_target_views // num_target_split]
            index_targets.append(target)
            index_unrolled.append(torch.arange(target[0]*self.cfg.temporal_downsample, target[-1]*self.cfg.temporal_downsample+self.cfg.temporal_downsample))
        index_targets = torch.concat(index_targets)
        index_unrolled = torch.concat(index_unrolled)
        if self.cfg.context_sampling == "uniform":
            context_indices = torch.linspace(0, extrinsics.shape[0] - 1, steps=num_context_views).long()

        elif self.cfg.context_sampling == "farthest_point":
            context_indices = fps_from_pose(extrinsics.float(), num_context_views)

        else:
            raise ValueError(f"Unknown context sampling strategy: {self.cfg.context_sampling}")
        

        if self.cfg.sample_cond_views:
            
            ref_idx = torch.randint(0, len(context_indices), size=(1, ))
            weights = torch.ones((extrinsics.shape[0], )).float()
            weights[context_indices[ref_idx]] = 0.0 # Ensure we don't sample the reference view again
            cond_indices = torch.multinomial(weights, self.cfg.max_cond_number - 1, replacement=False)
            cond_indices = torch.concat([context_indices[ref_idx], cond_indices])

        else:
            cond_indices = None

        return ViewIndex(context_indices, index_unrolled, cond=cond_indices), index_targets 
        
 

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views

    @property
    def min_context_views(self) -> int:
        return self.cfg.min_context_views
