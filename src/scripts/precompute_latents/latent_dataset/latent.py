import os
import torch
import numpy as np
import torchvision.transforms as tf

from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
from torch.utils.data import Dataset



@dataclass
class LatentDatasetCfg:
    ckpt_path: Path
    image_shape: list[int]
    data_root: Path
    min_frames: int
    downsample_factors: List

class LatentDataset(Dataset):
    def __init__(
        self, 
        cfg: LatentDatasetCfg, 
        stage: str, 
        index: int = 0, 
        size: int = None,
        output_dir: Path = Path(""), 
        flip: bool = False,
    ) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.stage = stage
        self.flip = flip
        
        self.output_dir = output_dir / stage
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        
        self.to_tensor = tf.ToTensor()
        
        self._init_data(stage, index, size)
        self._init_model()
    
    def _init_data(self, stage, index, size):
        raise NotImplementedError()
    
    def _init_model(self):
        raise NotImplementedError()
    def _get_data(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
         raise NotImplementedError()

    def save_data(self, downsample, scene, latents, extrinsics, intrinsics):
        # folder path ==> root / stage / downsample / scene[_flipped].npz
        out_path = self.output_dir / downsample
        os.makedirs(out_path, exist_ok=True) 
        np.savez(
            out_path / f"{scene}{'_flipped' if self.flip else ''}.npz",
            latent=latents.to(torch.float16).numpy(),
            extrinsics=extrinsics.to(torch.float16).numpy(),
            intrinsics=intrinsics.to(torch.float16).numpy(),
            allow_pickle=False
        )