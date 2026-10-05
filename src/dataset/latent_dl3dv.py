import os
import torch
import numpy as np

from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import Literal, Optional, Tuple, Union, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from src.misc.dl3dv_utils import load_metadata
from src.misc.camera_utils import rescale_and_crop, reflect_views, convert_poses
from src.scripts.precompute_latents.latent_dataset.latent import LatentDataset, LatentDatasetCfg

@dataclass
class LatentDL3DVDatasetCfg(LatentDatasetCfg):
    name: Literal["dl3dv"] = "dl3dv"
    num_workers: int = 8
    subset: str = ""
    encode_depth: bool=False
    min_depth: float = 0.616595
    max_depth: float = 645.6542

class LatentDL3DVDataset(LatentDataset):
    cfg: LatentDL3DVDatasetCfg

    def _init_data(self, stage, index, size, **kwargs):
        if stage == "train":
            root_chunks = sorted((self.cfg.data_root / stage / self.cfg.subset).glob("*"))
        else:
            # DL3DV test set. Newer releases nest scenes under
            # images/{scene}/{scene}/nerfstudio/ (each with images_4/ + transforms.json);
            # older layouts had {scene}/nerfstudio/ directly under the stage dir.
            root_chunks = sorted((self.cfg.data_root / stage / "images").glob("*/*/nerfstudio"))
            if not root_chunks:
                root_chunks = sorted((self.cfg.data_root / stage).glob("*/nerfstudio"))
        
        start = index * size
        end = (index + 1) * size
        self.chunks = root_chunks[start:end] if size else root_chunks
        print(f"(LatentDL3DVDataset): Number of workers={self.cfg.num_workers}")
        print(f"(LatentDL3DVDataset): Using chunk index range [{start}, {end}]")
    
    def _init_model(self):
        raise NotImplementedError
    
    def _load_one(self, chunk_path, img_path: str):
        with Image.open(chunk_path / "images_4" / img_path) as im:
            im = im.convert("RGB")
            im = np.asarray(im)
            im = torch.from_numpy(im)
            im = im.permute(2, 0, 1).contiguous()  # CHW uint8
        return im



    def load_images_to_tensor_threaded(
        self,
        chunk_path: Path,
        img_files: List[str],
        num_workers: int = 8,
    ) -> torch.Tensor:

        files = [str(f) for f in img_files]
        out = [None] * len(files)

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(self._load_one, chunk_path, f) for f in img_files]

            for i, fut in enumerate(tqdm(futures, leave=False, desc="Loading images")):
                out[i] = fut.result()

        return torch.stack(out, dim=0)
    
    def _get_data(self, idx):

        chunk_path = self.chunks[idx]
        images = []
        img_files = sorted(os.listdir(chunk_path / "images_4"))
        depths = None
        if self.cfg.encode_depth:
            depth_file = chunk_path / "depths.npz"
            depths = np.load(depth_file, allow_pickle=False)["depths"]
            depths = torch.from_numpy(depths).unsqueeze(1).expand(-1, 3, -1, -1)  # (V, 1, H, W)
            depths = (depths.clip(self.cfg.min_depth, self.cfg.max_depth) - self.cfg.min_depth) / (self.cfg.max_depth - self.cfg.min_depth)
                                 
        images = self.load_images_to_tensor_threaded(chunk_path, img_files, self.cfg.num_workers) 
        key = chunk_path.name
        if key == "nerfstudio":
            key = chunk_path.parent.name
        scene = key
        example = load_metadata(chunk_path / "transforms.json")
        extrinsics, intrinsics = convert_poses(example["cameras"])

        if images.shape[2] < 256 or images.shape[3] < 256:
            print(f"Skipping {scene} due to bad shape: {images.shape}")
            return
        scene = key
        # images = images.to(torch.float)
        images = images / 255.0
        old_intrinsics = intrinsics.clone()
        images, intrinsics = rescale_and_crop(images, old_intrinsics, tuple(self.cfg.image_shape))
        
        if self.cfg.encode_depth:
            depths, _ = rescale_and_crop(depths, old_intrinsics, tuple(self.cfg.image_shape))
        

        if self.flip:
            old_extrinsics = extrinsics.clone()
            images, extrinsics = reflect_views(images, old_extrinsics)
            if self.cfg.encode_depth:
                depths, _ = reflect_views(depths, old_extrinsics)

        return scene, images, depths, intrinsics, extrinsics

    def __getitem__(self, idx): 
         raise NotADirectoryError()

    def __len__(self):
        return len(self.chunks)