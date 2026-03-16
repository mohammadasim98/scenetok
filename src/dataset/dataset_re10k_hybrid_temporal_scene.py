import json
from dataclasses import asdict, dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal
import re
import torch
import torchvision.transforms as tf
import os
from torch.utils.data import IterableDataset
import json 
from .dataset import DatasetCfgCommon

from .dtypes import Stage
from .view_sampler import ViewSampler
import numpy as np

def rotation_log(R):
    """
    Compute the logarithm (axis-angle vector) of a batch of SO(3) rotation matrices.
    Args:
        R: (..., 3, 3) batch of rotation matrices
    Returns:
        (..., 3) axis-angle vectors
    """
    cos_theta = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1 + 1e-6, 1 - 1e-6)  # numerical stability
    theta = torch.acos(cos_theta)

    skew = 0.5 * (R - R.transpose(-2, -1))
    axis = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)
    axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-9)

    return axis * theta.unsqueeze(-1)

def pose_distance(pose1, pose2, w_pos=1.0, w_rot=1.0):
    """
    Computes distance matrix between two sets of poses in SE(3).
    pose1, pose2: (N, 4, 4)
    Returns: (N, N) matrix of distances
    """
    t1 = pose1[:, :3, 3]  # (N, 3)
    t2 = pose2[:, :3, 3]  # (N, 3)

    pos_dist = torch.cdist(t1, t2, p=2) ** 2  # (N, N)

    R1 = pose1[:, :3, :3]  # (N, 3, 3)
    R2 = pose2[:, :3, :3]  # (N, 3, 3)

    N = R1.shape[0]
    rel_rot = torch.matmul(R1.unsqueeze(1).transpose(-2, -1), R2.unsqueeze(0))  # (N, N, 3, 3)
    log_rot = rotation_log(rel_rot.view(-1, 3, 3)).view(N, N, 3)
    rot_dist = (log_rot ** 2).sum(dim=-1)  # (N, N)

    return w_pos * pos_dist + w_rot * rot_dist

def fps_from_pose(pose_matrices, n_samples, w_pos=1.0, w_rot=1.0):
    """
    Args:
        pose_matrices: (N, 4, 4) tensor of SE(3) matrices
        n_samples: number of poses to sample
    Returns:
        sorted_indices: (n_samples,) sorted indices of sampled poses
    """
    N = pose_matrices.shape[0]
    device = pose_matrices.device

    dmat = pose_distance(pose_matrices, pose_matrices, w_pos, w_rot)  # (N, N)
    selected = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)

    farthest = torch.randint(0, N, size=(1,), dtype=torch.long).item()

    for i in range(n_samples):
        selected[i] = farthest
        dist_to_farthest = dmat[farthest]
        distances = torch.minimum(distances, dist_to_farthest)
        farthest = torch.argmax(distances).item()

    return torch.sort(selected)[0]

@dataclass
class DatasetRE10kHybridTemporalSceneCfg(DatasetCfgCommon):
    name: Literal["re10k_hybrid_temporal_scene"]
    target_root: Path | None
    context_root: Path | None
    baseline_epsilon: float
    max_fov: float
    make_baseline: bool
    limit_frame_distance: list | None = None
    max_cond_number: int=5


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

class DatasetRE10kHybridTemporalScene(IterableDataset):
    cfg: DatasetRE10kHybridTemporalSceneCfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetRE10kHybridTemporalSceneCfg,
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

            # reconstruct full path
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
            if self.cfg.view_sampler.name != "video_va_videodc_bounded":
                context_indices = fps_from_pose(context_extrinsics, self.cfg.view_sampler.num_context_views)

            if view_indices.context is not None:
                context_indices = view_indices.context
                ctxt_extrinsics = context_extrinsics[context_indices]
                if ctxt_extrinsics.shape[0] == 2 and self.cfg.make_baseline:
                    a, b = ctxt_extrinsics[:, :3, 3]
                    scale = (a - b).norm()
                    if scale < self.cfg.baseline_epsilon:
                        print(
                            f"Skipped {scene} because of insufficient baseline "
                            f"{scale:.6f}"
                        )
                        continue
                    target_extrinsics[:, :3, 3] /= scale
                    context_extrinsics[:, :3, 3] /= scale
            

            if context_indices is not None:
                ctxt_latents = context_latents[context_indices]
                ctxt_extr = context_extrinsics[context_indices]
                ctxt_intr = context_intrinsics[context_indices]
                
                ref_idx = torch.randint(0, len(context_indices), size=(1, ))
                cond_indices = torch.multinomial(torch.ones((context_latents.shape[0], )).float(), self.cfg.max_cond_number - 1, replacement=False)
                cond_indices = torch.concat([context_indices[ref_idx], cond_indices])

                sample["context"] = {
                    "extrinsics": ctxt_extr,
                    "intrinsics": ctxt_intr,
                    "latent": ctxt_latents,
                    "index": context_indices
                }

                sample["cond"] = {
                    "extrinsics": context_extrinsics[cond_indices],
                    "intrinsics": context_intrinsics[cond_indices],
                    "latent": context_latents[cond_indices],
                    "index": cond_indices
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

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            # Load the root's index.
            with (self.cfg.root / data_stage / "index.json").open("r") as f:
                index = json.load(f)
            index = {k: Path(self.cfg.root / data_stage / v) for k, v in index.items()}

            # The constituent datasets should have unique keys.
            assert not (set(merged_index.keys()) & set(index.keys()))

            # Merge the root's index into the main index.
            merged_index = {**merged_index, **index}
        return merged_index

    # def __len__(self) -> int:
    #     if isinstance(self.view_sampler, ViewSamplerEvaluation):
    #         return self.view_sampler.total_samples
    #     return len(self.index.keys())
