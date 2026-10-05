from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from src.misc.camera_utils import convert_poses

from .dataset import DatasetCfgCommon
from .dtypes import Stage
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .shims.random_transform_shim import apply_random_transform_shim
from .view_sampler import ViewSampler


@dataclass
class DatasetRE10kHiResCfg(DatasetCfgCommon):
    name: Literal["re10k_hires"]
    root: Path | None
    annotations_root: Path | None
    baseline_epsilon: float
    max_fov: float
    make_baseline: bool


class DatasetRE10kHiRes(Dataset):
    cfg: DatasetRE10kHiResCfg
    stage: Stage
    view_sampler: ViewSampler
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetRE10kHiResCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        force_shuffle: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.force_shuffle = force_shuffle

        if cfg.root is None:
            raise Exception(
                "Root directory of dataset is not defined. Please specify in your argument as dataset.root=<path-to-root-directory>"
            )
        if cfg.annotations_root is None:
            raise Exception(
                "Annotations root directory is not defined. Please specify in your argument as dataset.annotations_root=<path-to-annotations-root-directory>"
            )

        self.data_root = cfg.root / stage
        self.annotations_stage_root = cfg.annotations_root / stage

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data stage directory does not exist: {self.data_root}")
        if not self.annotations_stage_root.exists():
            raise FileNotFoundError(
                f"Annotations stage directory does not exist: {self.annotations_stage_root}"
            )

        all_scene_ids = sorted(path.stem for path in self.annotations_stage_root.glob("*.txt"))

        # Match evaluation datasets that rely on a fixed scene list from the view sampler.
        if hasattr(view_sampler, "index") and getattr(view_sampler, "index"):
            requested_scene_ids = list(getattr(view_sampler, "index").keys())
            scene_ids = [scene_id for scene_id in requested_scene_ids if scene_id in all_scene_ids]
        else:
            scene_ids = all_scene_ids

        self.scenes = []
        for scene_id in scene_ids:
            scene_dir = self.data_root / scene_id
            npz_path = scene_dir / "data.npz"
            camera_path = self.annotations_stage_root / f"{scene_id}.txt"
            if scene_dir.exists() and npz_path.exists() and camera_path.exists():
                self.scenes.append((scene_id, npz_path, camera_path))

    def _parse_camera_file(self, camera_path: Path) -> tuple[list[str], torch.Tensor]:
        timestamps: list[str] = []
        cameras = []

        with open(camera_path, "r") as file:
            lines = file.readlines()

        # First line is a URL in the RE10K annotation format.
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 19:
                continue

            timestamp = parts[0]
            fx, fy, cx, cy = map(float, parts[1:5])
            w2c_flat = [float(x) for x in parts[7:19]]

            # Keep the same 18D camera layout used by the standard RE10K loader.
            cameras.append([fx, fy, cx, cy, 0.0, 0.0] + w2c_flat)
            timestamps.append(timestamp)

        if not cameras:
            raise ValueError(f"No valid camera entries in {camera_path}")

        return timestamps, torch.tensor(cameras, dtype=torch.float32)

    def __getitem__(self, idx):
        scene, npz_path, camera_path = self.scenes[idx]

        timestamps, cameras = self._parse_camera_file(camera_path)
        extrinsics, intrinsics = convert_poses(cameras)

        with np.load(str(npz_path), allow_pickle=False) as npz_data:
            images = []
            valid_indices = []
            for i, timestamp in enumerate(timestamps):
                key = f"{timestamp}.jpg"
                if key not in npz_data:
                    continue
                image = torch.from_numpy(npz_data[key]).permute(2, 0, 1).float() / 255.0
                images.append(image)
                valid_indices.append(i)

        if not images:
            raise ValueError(f"No valid images found for scene {scene}")

        images = torch.stack(images)
        valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.long)
        extrinsics = extrinsics[valid_indices_tensor]
        intrinsics = intrinsics[valid_indices_tensor]

        num_views = extrinsics.shape[0]
        print(f"Scene {scene}: {num_views} valid views found.")
        sampled, downsampled_indices = self.view_sampler.sample(
            num_views=num_views,
            num_latents=num_views,
            scene=scene,
            stage=self.stage, 
            extrinsics=extrinsics
        )

        if hasattr(sampled, "context"):
            view_indices_list = [sampled]
        elif isinstance(sampled, tuple):
            view_indices_list = [sampled[0]]
        else:
            view_indices_list = sampled
        
        for view_index in view_indices_list:
            sample = {"scene": scene}

            context_extrinsics = extrinsics[view_index.context]
            if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline:
                a, b = context_extrinsics[:, :3, 3]
                scale = (a - b).norm()
                if scale < self.cfg.baseline_epsilon:
                    print(
                        f"Skipped {scene} because of insufficient baseline {scale:.6f}"
                    )
                    continue
                extrinsics[:, :3, 3] /= scale

            for view_type, indices in asdict(view_index).items():
                if indices is None:
                    continue
                if view_type=="target":
                    indices = indices // 4
                sample[view_type] = {
                    "extrinsics": extrinsics[indices],
                    "intrinsics": intrinsics[indices],
                    "latent": images[indices],
                    "index": indices,
                }
            else:
                if self.stage == "train" and self.cfg.augment:
                    sample = apply_augmentation_shim(sample)
                if self.stage in ["train", "val"] and self.cfg.random_transform_extrinsics:
                    sample = apply_random_transform_shim(sample)
                return apply_crop_shim(sample, tuple(self.cfg.shape))

        raise RuntimeError(f"Could not sample valid views for scene {scene}")

    def __len__(self) -> int:
        return len(self.scenes)
