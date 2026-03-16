import torch
import numpy as np

from typing import Literal
from dataclasses import dataclass

from src.misc.camera_utils import fps_from_pose
from .view_sampler import ViewIndex, ViewSampler, ViewSamplerCfg
from ..dtypes import Stage

@dataclass
class ViewSamplerBoundedCfg(ViewSamplerCfg):
    name: Literal["bounded"]

    min_distance_between_context_views: int = 0
    max_distance_between_context_views: int | None = None
    max_distance_to_context_views: int = 0
    context_gap_warm_up_steps: int = 0
    target_gap_warm_up_steps: int = 0
    initial_min_distance_between_context_views: int = 0
    initial_max_distance_between_context_views: int | None = None
    initial_max_distance_to_context_views: int = 0

    num_target_split: int=1
    chunk_index_gap: int=1
    target_split_prob: float=0.0
    temporal_downsample: int=1
    offset: int=0


class ViewSamplerBounded(ViewSampler[ViewSamplerBoundedCfg]):
    def schedule(
        self, 
        initial: int, 
        final: int,
        steps: int
    ) -> int:
        fraction = self.global_step / steps

        
        return min(initial + int((final - initial) * fraction), final)
    
    def latent_to_original_index(self, latent_idx):

        if latent_idx % self.cfg.chunk_index_gap==0:
            idx = self.cfg.temporal_downsample  * latent_idx - (self.cfg.temporal_downsample-1)*(latent_idx//self.cfg.chunk_index_gap)
        else:
            idx = self.cfg.temporal_downsample  * latent_idx - (self.cfg.temporal_downsample-1)*(latent_idx//self.cfg.chunk_index_gap + self.cfg.offset)

        return idx

    def original_to_latent_index(self, idx):
        frame_index_gap = self.cfg.temporal_downsample * (self.cfg.chunk_index_gap-1) + 1
        k = idx // frame_index_gap
        r = idx % frame_index_gap
        return self.cfg.chunk_index_gap * k + (r + self.cfg.temporal_downsample-1) // self.cfg.temporal_downsample
    
    def sample(
        self,
        num_views: int,
        num_latents: int,
        stage: Stage,
        extrinsics: torch.Tensor,
        **kwargs
    ) -> list[ViewIndex]:
        offset = self.cfg.offset

        temporal_downsample = self.cfg.temporal_downsample
        max_distance_between_context_views = \
            self.cfg.max_distance_between_context_views or num_views
        initial_max_distance_between_context_views = \
            self.cfg.initial_max_distance_between_context_views or num_views
        # Compute the context view spacing based on the current global step.
        if self.stage == "test":
            # When testing, always use the full gap.
            max_context_gap = max_distance_between_context_views
            min_context_gap = max_distance_between_context_views
        elif self.cfg.context_gap_warm_up_steps > 0:
            max_context_gap = self.schedule(
                initial_max_distance_between_context_views,
                max_distance_between_context_views,
                self.cfg.context_gap_warm_up_steps
            )
        

            min_context_gap = self.schedule(
                self.cfg.initial_min_distance_between_context_views,
                self.cfg.min_distance_between_context_views,
                self.cfg.context_gap_warm_up_steps
            )
        else:
            max_context_gap = min(max_distance_between_context_views, num_views)
            min_context_gap = min(self.cfg.min_distance_between_context_views, num_views)

        if not self.cameras_are_circular:
            max_context_gap = min(max_context_gap, num_views - 1)
            min_context_gap = min(min_context_gap, num_views - 1)

        # if not self.cameras_are_circular:
        #     max_context_gap = min(num_views - 1, max_context_gap)   # NOTE fixed former bug here

        # Compute the margin from context window to target window based on the current global step
        if self.stage != "test" and self.cfg.target_gap_warm_up_steps > 0:
            max_target_gap = self.schedule(
                self.cfg.initial_max_distance_to_context_views,
                self.cfg.max_distance_to_context_views,
                self.cfg.target_gap_warm_up_steps
            )
        else:
            max_target_gap = self.cfg.max_distance_to_context_views
        # Pick the gap between the context views.
        if max_context_gap < min_context_gap:
            # max_context_gap = min_context_gap
            # context_gap = max_context_gap
            raise ValueError(f"Example does not have enough frames! {max_context_gap} <= f <= {min_context_gap}, and num views: {num_views}")
        elif max_context_gap == min_context_gap:
            context_gap = max_context_gap
        else:
            context_gap = torch.randint(
                min_context_gap,
                max_context_gap + 1,
                size=tuple()
            ).item()

        # Pick the left and right context indices.
        index_context_left = torch.randint(
            low=0,
            high=num_views if self.cameras_are_circular else num_views - context_gap,
            size=tuple()
        ).item()
        # if self.stage == "test":
        #     index_context_left = index_context_left * 0
        index_context_right = index_context_left + context_gap

        index_unrolled = None
        num_target_split = self.cfg.num_target_split if stage == "train" else 1
        # Compute target indices
        if self.cfg.num_target_views > 0:
            index_target_left = index_context_left - max_target_gap
            index_target_right = index_context_right + max_target_gap

            if not self.cameras_are_circular:
                index_target_left = max(0, index_target_left)
                index_target_right = min(num_views-1, index_target_right)
            

            # Pick the target view indices.

            num_target_views = self.cfg.num_target_views
            chunk_index_gap = self.cfg.chunk_index_gap
            # When training or validating (visualizing), pick at random.


            if offset == 0:
                num_latents = num_views // temporal_downsample
                start = ((index_target_left // temporal_downsample) // chunk_index_gap) * chunk_index_gap
                end = ((index_target_right // temporal_downsample) // chunk_index_gap) * chunk_index_gap

            else:
                # print(num_views, num_latents)
                # print(index_target_left, index_target_right)
                start = self.original_to_latent_index(index_target_left)
                end = self.original_to_latent_index(index_target_right)
                # print(start, end)
                start = (start // chunk_index_gap) * chunk_index_gap
                end = (end // chunk_index_gap) * chunk_index_gap
                # print(start, end)
            try:
                starting_indices = torch.arange(start, end - num_target_views + 1, chunk_index_gap)
            except:
                print(start, end)
                print(max_context_gap, min_context_gap)
                print(index_context_left, index_context_right)
                print(num_views, num_latents)
                raise ValueError("Error in generating starting indices")
            # print(starting_indices)
            num_target_split = min(len(starting_indices), num_target_split)
            index_target = torch.arange(0, num_latents).long()
            if np.random.choice([True, False], size=1, p=[1 - self.cfg.target_split_prob, self.cfg.target_split_prob]):
                num_target_split = 1
            
            idxs = torch.multinomial(torch.ones_like(starting_indices).float(), num_target_split, replacement=False)
            index_targets = []
            index_unrolled = []
            # print(index_target)
            for idx in idxs:
                starting_index = starting_indices[idx]
                # print(starting_index, starting_index + num_target_views // num_target_split)
                target = index_target[starting_index:starting_index + num_target_views // num_target_split]
                index_targets.append(target)

                if offset == 0:
                    idx_unrolled = torch.arange(target[0]*temporal_downsample, target[-1]*temporal_downsample+temporal_downsample)
                else:
                    try:

                        start = self.latent_to_original_index(target[0])
                    except IndexError as err:
                        print("Num views: ", num_views)
                        print("Num latents: ", num_latents)
                        print("Starting Index list: ", starting_indices)
                        print("Selected Index list: ", idxs)
                        print("Target list: ", target)
                        print("Error: ", err)
                        return self.sample(num_views=num_views, num_latents=num_latents)


                    try:

                        end = self.latent_to_original_index(target[-1]+1)
                    except IndexError as err:
                        print("Starting Index list: ", starting_indices)
                        print("Selected Index list: ", idxs)
                        print("Target list: ", target)
                        print("Error: ", err)
                        return self.sample(num_views=num_views, num_latents=num_latents)

                    idx_unrolled = torch.arange(start, end)
                
                index_unrolled.append(idx_unrolled)
            index_target = torch.concat(index_targets)
            index_unrolled = torch.concat(index_unrolled)
            if len(index_unrolled) < 34 and offset==1:
                print(num_views, num_latents)
                print(start, end)
                print(starting_index, starting_index + num_target_views // num_target_split)
                print("Starting Index list: ", starting_indices)
                print("Selected Index list: ", idxs)
                print("Target list: ", index_target)
        else:
            index_target = None

        indices = []
        if self.cfg.num_context_views > 2:
            if self.cfg.context_sampling == "uniform":
                context_indices = torch.linspace(index_context_left, index_context_right + 1, steps=self.cfg.num_context_views).long()
                indices = context_indices[1:-1].tolist()
                index_context_left = context_indices[0].item()
                index_context_right = context_indices[-1].item()

            elif self.cfg.context_sampling == "farthest_point":
                context_indices = torch.arange(0, num_views).long()
                fps_indices = fps_from_pose(extrinsics[index_context_left:index_context_right+1].float(), n_samples=self.cfg.num_context_views).tolist()
                indices = context_indices[index_context_left:index_context_right+1][fps_indices[1:-1]].tolist()

            else:
                raise ValueError(f"Unknown context sampling strategy: {self.cfg.context_sampling}")
        # Apply modulo for circular datasets.
        if self.cameras_are_circular:
            if index_target is not None:
                index_target %= num_views
            index_context_right %= num_views

        return ViewIndex(torch.tensor(sorted([index_context_left, *indices, index_context_right])), index_unrolled), index_target

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
