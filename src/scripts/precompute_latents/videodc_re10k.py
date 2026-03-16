

import torch

from tqdm import tqdm
from typing import Literal
from dataclasses import dataclass

from opensora.models.dc_ae import DC_AE
from src.scripts.precompute_latents.latent_dataset import LatentRE10KDataset, LatentRE10KDatasetCfg

@dataclass
class LatentVideoDCRE10KDatasetCfg(LatentRE10KDatasetCfg):
    name: Literal["videodc_re10k"]

class LatentVideoDCRE10KDataset(LatentRE10KDataset):
    cfg: LatentVideoDCRE10KDatasetCfg

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

        for i, (scene, images, intrinsics, extrinsics) in enumerate(tqdm(self._get_data(idx))):

            total_views = images.shape[0]
            if images.shape[0] != extrinsics.shape[0]:
                print(f"Mismatch in number of images and extrinsics: {images.shape[0]} != {extrinsics.shape[0]}")
                continue
            if total_views < self.cfg.min_frames:
                print(f"Total number of frames {total_views} < minimum frame counts {self.cfg.min_frames}")
                continue
            
            # May want to downsample frames which requires to compute latents separately for each downsampling factor
            downsamples = self.cfg.downsample_factors
            for downsample in downsamples:
                downsampled_total_views = total_views// downsample
                if downsampled_total_views < self.cfg.min_frames :
                    print(f"Downsampled number of frames {downsampled_total_views} < minimum frame counts {self.cfg.min_frames}")
                    continue
                
                imgs = images[::downsample]
                extr = extrinsics[::downsample]
                intr = intrinsics[::downsample]
                if imgs.shape[0] != extr.shape[0]:
                    print(f"Mismatch in factor {downsample} in number of images and extrinsics: {imgs.shape[0]} != {extr.shape[0]}")
                    continue
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
                    continue
                
                self.save_data(str(downsample), scene, latents, extr, intr)

