
import math
import torch

from tqdm import tqdm
from typing import Literal
from dataclasses import dataclass


from src.model.autoencoder.vavae.vavae import AutoencoderVA
from src.scripts.precompute_latents.latent_dataset import LatentDL3DVDataset, LatentDL3DVDatasetCfg


@dataclass
class LatentVADL3DVDatasetCfg(LatentDL3DVDatasetCfg):
    name: Literal["va_dl3dv"]
    encode_batch_size: int=16

class LatentVADL3DVDataset(LatentDL3DVDataset):

    cfg: LatentVADL3DVDatasetCfg

    def _init_model(self):
        
        self.model = AutoencoderVA(None).from_pretrained(self.cfg.ckpt_path).cuda().eval()
    
    
    def __getitem__(self, idx):

        scene, images, intrinsics, extrinsics = self._get_data(idx)
            
        total_views = images.shape[0]
        if images.shape[0] != extrinsics.shape[0]:
            print(f"Mismatch in number of images and extrinsics: {images.shape[0]} != {extrinsics.shape[0]}")
            return
        
        if total_views < self.cfg.min_frames:
            # print(f"Total number of frames {total_views} < minimum frame counts {self.cfg.min_frames}")
            return
        
        
        # Process in a batch of 16 to avoid OOM (Adjust as per your need)
        latents = []
        for batch in tqdm(torch.split(images, split_size_or_sections=self.cfg.encode_batch_size), leave=False, desc="Running chunks"):
            with torch.no_grad():
                latent = self.model.encode(2 * batch.cuda() - 1).latent_dist.sample()
            
            if torch.isnan(latent).sum():
                print("nan is detected")
                exit()
            
            latents.append(latent.cpu().float())
        latents = torch.cat(latents).contiguous()

        self.save_data("1", scene, latents, extrinsics, intrinsics)
