import gc

import torch
from lpips import LPIPS

from torchmetrics.functional.image import structural_similarity_index_measure
from torch import nn
from skimage.metrics import structural_similarity
from cleanfid import fid

import numpy as np
from scipy import linalg
from tqdm import tqdm
from einops import rearrange, reduce
import torch
from torch.nn.functional import interpolate
from torchmetrics.image.fid import FrechetInceptionDistance
from ..misc.torch_utils import convert_to_buffer
from submodules.fvd.frechet_video_distance import frechet_video_distance

def freeze(m: torch.nn.Module) -> None:
    for param in m.parameters():
        param.requires_grad = False
    m.eval()


class Metric(nn.Module):
    def __init__(self):
        """
        Args:
            config (MultiValConfig): Hydra config with validation parameters.
            dataloader_map (dict): Mapping from name to an actual DataLoader.
        """
        super().__init__()
        self.setup()

    def setup(self):

        self.lpips = LPIPS(net="vgg").eval().to(torch.float16)
        # self.fid = FrechetInceptionDistance(normalize=True, feature_extractor_weights_path="checkpoints/pt_inception-2015-12-05-6726825d.pth").eval()
        self.fid = FrechetInceptionDistance(
            feature=2048,
            sync_on_compute=False,
            dist_sync_on_step=False,
            compute_on_cpu=True).eval()
        self.pred_list = []
        self.gt_list = []
        # freeze(self.lpips)
        # convert_to_buffer(self.lpips, persistent=False)

        # freeze(self.fid)
        # convert_to_buffer(self.fid, persistent=False)
        # self.fid = FrechetInceptionDistance(normalize=True).eval().to(torch.bfloat16)


    def compute_mse(self, pred, gt, num_views: int=16):
        return ((pred - gt)**2).mean()
    
    def compute_psnr(self, pred, gt, num_views: int=16):

        gt = gt.clip(min=0, max=1)
        pred = pred.clip(min=0, max=1)
        mse = reduce((gt - pred) ** 2, "b c h w -> b", "mean")
        return (-10 * mse.log10()).mean()

    @torch.no_grad()
    def compute_lpips(self, pred, gt, num_views: int=16):
        if pred.ndim == 3:
            pred = pred[None]
        if gt.ndim == 3:
            gt = gt[None]
        
        l = 0
        for p, g in zip(pred, gt):
            l += self.lpips(p[None].to(torch.float16), g[None].to(torch.float16), normalize=True).to(pred.dtype)
        return (l / pred.shape[0]).mean()
    
    def compute_ssim(self, pred, gt, num_views: int=16):
        if pred.ndim == 3:
            pred = pred[None]
        if gt.ndim == 3:
            gt = gt[None]
        ssim = [
            structural_similarity(
                gt.float().detach().cpu().numpy(),
                hat.float().detach().cpu().numpy(),
                win_size=11,
                gaussian_weights=True,
                channel_axis=0,
                data_range=1.0,
            )
            for gt, hat in zip(gt, pred)
        ]

        return torch.tensor(ssim, dtype=pred.dtype, device=pred.device).mean()
    
    def update_fid(self, pred, gt, num_views: int=16):
        pred = (pred * 255).to(torch.uint8)
        gt = (gt * 255).to(torch.uint8)
        pred = rearrange(pred, "(b v) c h w -> b v c h w", v=num_views)
        gt = rearrange(gt, "(b v) c h w -> b v c h w", v=num_views)
        for i in range(pred.shape[0]):   # or even 8
            self.fid.update(gt[i].to("cuda:0"), real=True)
            self.fid.update(pred[i].to("cuda:0"), real=False)
    
    def update_fvd(self, pred, gt, num_views: int=16):

        pred = rearrange(pred, "(b v) c h w -> b v h w c", v=num_views)
        gt = rearrange(gt, "(b v) c h w -> b v h w c", v=num_views)
        pred = (pred * 255).float()
        gt = (gt * 255).float()

        self.pred_list.append(pred.cpu())
        self.gt_list.append(gt.cpu())

    def reset_fid(self):
        self.fid.reset()
    def reset_fvd(self):
        self.pred_list = []
        self.gt_list = []
    @torch.no_grad()
    def compute_fid(self, pred=None, gt=None, update=True, num_views: int=16):
        
        if update:
            self.update_fid(pred, gt)
        
        fid = self.fid.compute()

        return fid
    
    @torch.no_grad()
    def compute_fvd(self, pred=None, gt=None, update=True, num_views: int=16):

        if update:
            self.update_fvd(pred, gt)
        
        pred_tensor = torch.cat(self.pred_list, dim=0)
        gt_tensor = torch.cat(self.gt_list, dim=0)

        fvd = frechet_video_distance(pred_tensor, gt_tensor, "submodules/fvd/pytorch_i3d_model/models/rgb_imagenet.pt")

        return torch.tensor(fvd)

    @torch.no_grad()
    def forward(self, pred, gt, num_views: int=16, **kwargs):
        _dict = {}

        for key in list(kwargs.keys()):
            if kwargs.get(key, False):
                _dict[key] = getattr(self, f"compute_{key}")(pred, gt, num_views=num_views)
                gc.collect()
                torch.cuda.empty_cache()

        return _dict



