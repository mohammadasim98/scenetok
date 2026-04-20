
import torch

from tqdm import tqdm
from typing import Literal
from dataclasses import dataclass

from src.model.autoencoder.wanvae import Wan2_2_VAE
from src.scripts.precompute_latents.latent_dataset import LatentRE10KDataset, LatentRE10KDatasetCfg
from src.model.autoencoder.autoencoder_wan import AutoencoderWan, AutoencoderWanCfg, WanKwargsCfg

@dataclass
class LatentWanRE10KDatasetCfg(LatentRE10KDatasetCfg):
    name: Literal["wan_re10k"]


class LatentWanRE10KDataset(LatentRE10KDataset):
    cfg: LatentWanRE10KDatasetCfg

    def _init_model(self):
        print(f"Loading Wan2.2 VAE model from {self.cfg.ckpt_path}...")
        # Encode as multiples of 17 frames (causal 4× temporal downsampling: 17 → 5 latents)
        # self.model = Wan2_2_VAE(
        #     vae_pth=str(self.cfg.ckpt_path)
        # ).cuda().eval()

        self.model = AutoencoderWan(AutoencoderWanCfg(
            name="wan",
            pretrained_from=str(self.cfg.ckpt_path),
            kwargs=WanKwargsCfg(
                in_channels=3,
                latent_channels=48,
                scaling_factor=1.0
            )
        )).from_pretrained(str(self.cfg.ckpt_path)).cuda().to(torch.bfloat16).eval()

    def __getitem__(self, idx):

        for i, (scene, images, intrinsics, extrinsics) in enumerate(tqdm(self._get_data(idx))):

            total_views = images.shape[0]
            if images.shape[0] != extrinsics.shape[0]:
                print(f"Mismatch in number of images and extrinsics: {images.shape[0]} != {extrinsics.shape[0]}")
                continue
            if total_views < self.cfg.min_frames:
                print(f"Total number of frames {total_views} < minimum frame counts {self.cfg.min_frames}")
                continue

            downsamples = self.cfg.downsample_factors
            for downsample in downsamples:
                downsampled_total_views = total_views // downsample
                if downsampled_total_views < self.cfg.min_frames:
                    print(f"Downsampled number of frames {downsampled_total_views} < minimum frame counts {self.cfg.min_frames}")
                    continue

                imgs = images[::downsample]
                extr = extrinsics[::downsample]
                intr = intrinsics[::downsample]
                if imgs.shape[0] != extr.shape[0]:
                    print(f"Mismatch in factor {downsample} in number of images and extrinsics: {imgs.shape[0]} != {extr.shape[0]}")
                    continue

                v, c, h, w = imgs.shape

                # Wan2.2 VAE requires input in chunks of 17 frames
                num = (v // 17) * 17
                if num == 0:
                    print(f"Not enough frames for Wan encoding ({v} frames), need at least 17")
                    continue

                # (c, num, h, w), normalized to [-1, 1]
                imgs_enc = imgs[:num].cuda().to(torch.bfloat16)
                imgs_enc = 2 * imgs_enc - 1.0
                extr = extr[:num]
                intr = intr[:num]

                chunks = num // 17
                imgs_list = list(torch.chunk(imgs_enc, chunks, dim=0))  # list of (17, c, h, w)

                with torch.no_grad():
                    latents = []
                    for img in imgs_list:
                        latents.append(self.model.encode(img[None]))  # list of (t_latent, c, h_latent, w_latent)

                latents = torch.concat(latents, dim=1).cpu()  # (v_latent, c, h, w)

                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    print("Found NaNs in latents *before* saving! Skipping...")
                    continue

                self.save_data(str(downsample), scene, latents, extr, intr)
