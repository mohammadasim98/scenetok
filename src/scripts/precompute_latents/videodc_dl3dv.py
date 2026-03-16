

import torch

from tqdm import tqdm
from typing import Literal
from dataclasses import dataclass

from opensora.models.dc_ae import DC_AE
from src.scripts.precompute_latents.latent_dataset import LatentDL3DVDataset, LatentDL3DVDatasetCfg

@dataclass
class LatentVideoDCDL3DVDatasetCfg(LatentDL3DVDatasetCfg):
    name: Literal["videodc_dl3dv"]

class LatentVideoDCDL3DVDataset(LatentDL3DVDataset):
    cfg: LatentVideoDCDL3DVDatasetCfg

    def _init_model(self):
        print(f"Loading DC-AE model from {self.cfg.ckpt_path}...")

        # Encode as multiple of 16 in terms of number of views or multiple of 4 in terms of number of latent
        temporal_tile_size = 16
        tile_overlap_factor = 0.0
        self.model = DC_AE(
            model_name="dc-ae-f32t4c128",
            from_pretrained=str(self.cfg.ckpt_path),
            from_scratch=True,
            use_spatial_tiling=True,
            use_temporal_tiling=True,
            spatial_tile_size=256,
            temporal_tile_size=temporal_tile_size,
            tile_overlap_factor=tile_overlap_factor,
        ).to("cuda", dtype=torch.bfloat16).eval()
        self.model.temporal_tile_size = temporal_tile_size
        self.model.temporal_tile_latent_size = temporal_tile_size // 4

    def __getitem__(self, idx):

        scene, images, intrinsics, extrinsics = self._get_data(idx)

        total_views = images.shape[0]
        if images.shape[0] != extrinsics.shape[0]:
            print(f"Mismatch in number of images and extrinsics: {images.shape[0]} != {extrinsics.shape[0]}")
            return
        if total_views < self.cfg.min_frames:
            print(f"Total number of frames {total_views} < minimum frame counts {self.cfg.min_frames}")
            return
        
        # May want to downsample frames which requires to compute latents separately for each downsampling factor
        downsamples = self.cfg.downsample_factors
        for downsample in downsamples:
            downsampled_total_views = total_views// downsample
            if downsampled_total_views < self.cfg.min_frames :
                print(f"Downsampled number of frames {downsampled_total_views} < minimum frame counts {self.cfg.min_frames}")
                return
            
            imgs = images[::downsample]
            extr = extrinsics[::downsample]
            intr = intrinsics[::downsample]

            v, c, h, w = imgs.shape
            

            new_v = (v // 4) * 4
            imgs = imgs[:new_v , ...]
            extr = extr[:new_v , ...]
            intr = intr[:new_v , ...]
            
            with torch.no_grad():
                latents = self.model.encode(2 * imgs.permute(1, 0, 2, 3)[None].cuda().to(torch.bfloat16) - 1)

            latents = latents[0].permute(1, 0, 2, 3).cpu()
            
            if torch.isnan(latents).any() or torch.isinf(latents).any():
                print("Found NaNs in latents *before* saving! Skipping...")
                return
            
            self.save_data(str(downsample), scene, latents, extr, intr)

