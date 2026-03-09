
import os
import torch
import numpy as np
import torchvision.transforms as tf

from PIL import Image
from pathlib import Path
from typing import Literal
from torch.utils.data import Dataset
from dataclasses import asdict, dataclass

from .dataset import DatasetCfgCommon
from .dtypes import Stage
from .view_sampler import ViewSampler
from src.misc.dl3dv_utils import load_metadata
from src.misc.camera_utils import rescale_and_crop, reflect_views, convert_poses

from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim

@dataclass
class DatasetDL3DVCfg(DatasetCfgCommon):
    name: Literal["dl3dv"]
    root: Path
    flip: bool=False
    scale_focal_by_256: bool=False # Allow compatibility with va-videodc, refer to docs/KNOWN_BUG.md
    scale_context_focal_by_256: bool=False # Allow compatibility with va-wan, refer to docs/KNOWN_BUG.md
    folder_key: str="images_4"

class DatasetDL3DV(Dataset):
    cfg: DatasetDL3DVCfg
    stage: Stage
    view_sampler: ViewSampler
    to_tensor: tf.ToTensor
    chunks: list[Path]

    def __init__(
        self,
        cfg: DatasetDL3DVCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        force_shuffle: bool = False
    ) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self.root = cfg.root / self.data_stage
        # Collect chunks.
        self.chunks = []

        self.chunks = sorted(
            [path.name for path in self.root.iterdir() if path.suffix != ".csv"]
        )

    def __getitem__(self, idx):
        chunk_name = self.chunks[idx]
        chunk_path = self.root / chunk_name
        if os.path.exists(chunk_path / "nerfstudio"):
            chunk_path = chunk_path / "nerfstudio"
        images = []
        for img_path in sorted(os.listdir(chunk_path / self.cfg.folder_key)):
            im = Image.open(chunk_path / self.cfg.folder_key / img_path).convert("RGB")
            im = np.asarray(im) / 255
            im = torch.from_numpy(im)
            im = im.permute(2, 0, 1)
            images.append(im)
        images = torch.stack(images)
        scene = chunk_name
        example = load_metadata(chunk_path / "transforms.json", scale_focal_by_256=self.cfg.scale_focal_by_256)
        extrinsics, intrinsics = convert_poses(example["cameras"])

        if images.shape[2] < 256 or images.shape[3] < 256:
            print(f"Skipping {scene} due to bad shape: {images.shape}")
            return

        images, intrinsics = rescale_and_crop(images, intrinsics, tuple(self.cfg.shape))
        
        flip = self.cfg.flip
        if self.cfg.augment and self.stage == "train":
            flip = np.random.choice([False, True]).astype(int)
        
        if flip:
            images, extrinsics = reflect_views(images, extrinsics)
        
        num_views = extrinsics.shape[0]

        view_indices, upsampled_indices = self.view_sampler.sample(
            num_views=num_views, 
            num_latents=num_views, 
            stage=self.stage, 
            extrinsics=extrinsics, 
            scene=scene
        )        
        sample = {"scene": scene}
        for view_type, indices in asdict(view_indices).items():
            if indices is None:
                continue
            
                
            sample[view_type] = {
                "extrinsics": extrinsics[indices],
                "intrinsics": intrinsics[indices],
                "latent": images[indices],
                "index": indices
            }

            if view_type == "context" and self.cfg.scale_context_focal_by_256:
                sample[view_type]["intrinsics"][..., 0, 0] *= 3840/256
                sample[view_type]["intrinsics"][..., 1, 1] *= 2160/256
                

        return sample
    
        
    @property
    def data_stage(self) -> Stage:

        if self.stage == "val":
            return "test"
        return self.stage

    def __len__(self) -> int:
        return len(self.chunks)
   