import torch
import numpy as np
import torchvision.transforms as tf

from PIL import Image
from tqdm import tqdm
from io import BytesIO
from pathlib import Path
from typing import Literal, List
from dataclasses import dataclass, field
from torch.utils.data import Dataset

from src.misc.camera_utils import rescale_and_crop, reflect_views, convert_images, convert_poses
from src.scripts.precompute_latents.latent_dataset.latent import LatentDataset, LatentDatasetCfg

@dataclass
class LatentRE10KDatasetCfg(LatentDatasetCfg):
    name: Literal["re10k"]

class LatentRE10KDataset(LatentDataset):
    cfg: LatentRE10KDatasetCfg

    def _init_data(self, stage, index, size, **kwargs):
        root_chunks = sorted((self.cfg.data_root / stage).glob("*.torch"))
        self.chunks = root_chunks[index * size:(index + 1) * size] if size else root_chunks

    def _init_model(self):
        raise NotImplementedError
    
    def _get_data(self, idx):

        # Each chunk corresponds to individual *.torch chunk as defined by pixelsplat format for re10k
        chunk = torch.load(self.chunks[idx], weights_only=True)
        for i, data in enumerate(tqdm(chunk, leave=False, desc=f"Processing scenes from chunk {idx}")):
            scene = data["key"]
            extrinsics, intrinsics = convert_poses(data["cameras"])
            images = convert_images(data["images"], preprocess_fn=self.to_tensor)

            if images.shape[2] < 256 or images.shape[3] < 256:
                print(f"Skipping {scene} due to bad shape: {images.shape}")
                continue

            images, intrinsics = rescale_and_crop(images, intrinsics, tuple(self.cfg.image_shape))

            if self.flip:
                images, extrinsics = reflect_views(images, extrinsics)
            
            yield scene, images, intrinsics, extrinsics

    def __getitem__(self, idx): 
         raise NotADirectoryError()

    def __len__(self):
        return len(self.chunks)