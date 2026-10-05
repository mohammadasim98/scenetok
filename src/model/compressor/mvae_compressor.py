

import math
import torch

from dataclasses import dataclass, field
from torch import nn
from functools import reduce as re
from einops import rearrange
from typing import Literal, Optional
from pathlib import Path
from typing import Literal, Optional, Union, Tuple
from torch.utils.checkpoint import checkpoint

# Custom imports
from src.misc.torch_utils import pop_state_dict_by_prefix
from src.model.types import CompressorInputs
from src.model.camera import CameraCfg, get_camera
from src.model.encodings.embeddings import RotaryEmbedding3D
from src.model.diagonal_gaussian import DiagonalGaussianDistribution
from src.model.compressor.compressor import Compressor
from src.model.compressor.layers.attention import LightningDiTBlock
from src.model.denoiser.layers.pos_embed import VisionRotaryEmbeddingFast
from src.model.denoiser.layers.patch_embed import PatchEmbed 

def zero_initialize(layer):
    if hasattr(layer, 'weight') and layer.weight is not None:
        nn.init.zeros_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        nn.init.zeros_(layer.bias)
MODEL_PREFIX = "compressor."


@dataclass
class AggregatorKwargsCfg:
    hidden_size: int=768
    depth: int=12
    patch_size: int=1
    num_heads: int=12
    mlp_ratio: float=4.0
    use_qknorm: bool=False
    use_swiglu: bool=True
    use_rope_3d: bool=True
    use_rope_2d: bool=False
    use_rmsnorm: bool=True
    wo_shift: bool=False

@dataclass
class MVAECompressorCfg:
    name: Literal["mvae_compressor"]
    camera: CameraCfg
    kwargs: AggregatorKwargsCfg
    input_shape: Union[Tuple[int], list[int]]= (8, 8)
    token_dim: int=32
    num_scene_tokens: int=256
    use_ray_map: bool=True
    freeze_backbone: bool=True
    reproject_out: bool=True
    use_backbone: bool=False
    camera_conditioning: Literal["concat", "add", "adaLN"] = "concat"
    scene_token_projection: Literal["simple", "gated", "kl"] = "simple"
    ckpt_path: Optional[Path | str] = None
    load_strict: bool=True
    causal_self_atten: bool=False
    zero_init: bool=False
    gradient_checkpointing: bool=False
    init_large: bool=True
    mask_context: bool=False
    kl_schedule: list = field(default_factory= lambda: [100000, 300000])
    kl_weights: list = field(default_factory= lambda: [1e-10, 1e-7])
    norm_before_proj: bool=False
    init_small: bool=False
    use_scene_norm: bool=False
    freeze_after: int = -1  # Number of optimization steps after which to freeze the compressor (if -1, never freeze)
    unfreeze_after: int = -1
    freeze_q_after: int = -1  # Number of optimization steps after which to freeze the compressor (if -1, never freeze)
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, num_pos_feats):
        # x = tensor_list.tensors  # [B, C, H, W]
        # mask = tensor_list.mask  # [B, H, W], input with padding, valid as 0
        b, c, h, w = x.size()
        mask = torch.ones((b, h, w), device=x.device)  # [B, H, W]
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def freeze(m: nn.Module) -> None:
    for param in m.parameters():
        param.requires_grad = False
    m.eval()


def unfreeze(m: nn.Module) -> None:
    for param in m.parameters():
        param.requires_grad = True
    m.train()



class MVAECompressor(Compressor[MVAECompressorCfg]):

    def __init__(
        self,
        cfg: MVAECompressorCfg,
        in_channels: int,
        num_views: int,
        temporal_downsample: int=1,
        **kwargs
        ):

        super().__init__(cfg)
                
        self.scene_tokens = nn.Parameter(torch.randn(1, cfg.num_scene_tokens, cfg.kwargs.hidden_size))
        if cfg.zero_init:
            torch.nn.init.zeros_(self.scene_tokens)

        elif cfg.init_large:
            torch.nn.init.normal_(self.scene_tokens, std=5.0)
        elif cfg.init_small:
            torch.nn.init.normal_(self.scene_tokens, std=0.02)


        self.norm_before_proj = cfg.norm_before_proj
        if cfg.camera_conditioning == "concat":
            in_channels += 6
            cam_emb_dim = 6
        elif cfg.camera_conditioning == "add":
            cam_emb_dim = cfg.kwargs.hidden_size
        elif cfg.camera_conditioning == "adaLN":
            cam_emb_dim = cfg.kwargs.hidden_size
        else:
            raise NotImplementedError(f"Camera conditioning type {cfg.camera_conditioning} is invalid!")

        self.y_embed = PatchEmbed(cfg.input_shape, cfg.kwargs.patch_size, in_channels, cfg.kwargs.hidden_size)
        print("(Compressor) Using patch size: ", cfg.kwargs.patch_size)
        print("(Compressor) Using scene token: ", cfg.num_scene_tokens)
        print("(Compressor) Using scene token channels: ", cfg.token_dim if cfg.reproject_out else None)
        print("(Compressor) Using 3D RoPE: ", cfg.kwargs.use_rope_3d)
        print("(Compressor) Using 2D RoPE: ", cfg.kwargs.use_rope_2d)
        if cfg.kwargs.use_rope_3d:
            full_head_dim = cfg.kwargs.hidden_size // cfg.kwargs.num_heads
            self.feat_rope = RotaryEmbedding3D(full_head_dim, sizes=(num_views, cfg.input_shape[0] // cfg.kwargs.patch_size, cfg.input_shape[1] // cfg.kwargs.patch_size))
        elif cfg.kwargs.use_rope_2d:
            half_head_dim = cfg.kwargs.hidden_size // cfg.kwargs.num_heads // 2
            hw_seq_len = cfg.input_shape[0] // cfg.kwargs.patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None
        self.aggregate = nn.ModuleList([
            LightningDiTBlock(
                cfg.kwargs.hidden_size, 
                cfg.kwargs.num_heads, 
                mlp_ratio=cfg.kwargs.mlp_ratio, 
                use_qknorm=cfg.kwargs.use_qknorm, 
                use_swiglu=cfg.kwargs.use_swiglu, 
                use_rmsnorm=cfg.kwargs.use_rmsnorm,
                wo_shift=cfg.kwargs.wo_shift,
                is_rope_3d=cfg.kwargs.use_rope_3d,
                use_adaln=True if cfg.camera_conditioning == "adaLN" else False,
                skip_last_layer=True if i == cfg.kwargs.depth - 1 else False # Important if you want to enable DDP otherwise you need to fallback to ddp_find_unused_oarameters_true
            ) for i in range(cfg.kwargs.depth)
        ])
        self.use_scene_norm = cfg.use_scene_norm
        if cfg.use_scene_norm:
            self.scene_norm = nn.RMSNorm(cfg.kwargs.hidden_size)
        
        self.pose_embed = get_camera(cfg.camera, embed_dim=cam_emb_dim, temporal_downsample=temporal_downsample)
        self.causal = cfg.causal_self_atten
        
        if cfg.reproject_out:
            self.out_proj = nn.Linear(cfg.kwargs.hidden_size, cfg.token_dim)

    
        if cfg.reproject_out:
            if cfg.scene_token_projection == "simple":
                self.out_proj = nn.Linear(cfg.kwargs.hidden_size, cfg.token_dim)
                if self.norm_before_proj:
                    self.norm = nn.RMSNorm(cfg.kwargs.hidden_size)
                else:
                    self.norm = nn.RMSNorm(cfg.token_dim)
            elif cfg.scene_token_projection == "gated":
                self.out_proj = nn.Sequential(
                    nn.Linear(cfg.kwargs.hidden_size, cfg.token_dim),
                    nn.RMSNorm(cfg.token_dim),
                    nn.Tanh()
                )
            elif cfg.scene_token_projection == "kl":
                self.out_proj = nn.Sequential(
                    nn.Linear(cfg.kwargs.hidden_size, cfg.token_dim * 2),
                )
        if cfg.ckpt_path is not None:
            self.load_weights(cfg.ckpt_path, strict=cfg.load_strict)

    def load_weights(
        self,
        path: Path | str,
        **kwargs
    ):  
        print(f"(Compressor) Loading weights (strict={self.cfg.load_strict}) from: ", path)
        weights = torch.load(path, map_location=torch.device("cpu"))
 
        weights = pop_state_dict_by_prefix(weights["state_dict"], MODEL_PREFIX)

        with torch.no_grad():
            scene_tokens = weights["scene_tokens"]
            self.scene_tokens[:, :scene_tokens.shape[1]] = scene_tokens
            if scene_tokens.shape[1] != self.scene_tokens.shape[1]:
                print(f"Partially initialized scene tokens: {scene_tokens.shape} -> {self.scene_tokens.shape}")
            else:
                print(f"Fully initialized scene tokens: {scene_tokens.shape} -> {self.scene_tokens.shape}")
    
        if not self.cfg.load_strict:
            for key in list(weights.keys()):
                try:
                    param_shape = re(getattr, [self, *key.split(".")]).shape
                    if param_shape != weights[key].shape:
                        del weights[key]
                except AttributeError:
                    continue


        self.load_state_dict(weights, **kwargs)
    

    @property
    def output_dim(self) -> int:

        if self.cfg.reproject_out:
            cond_dim = self.cfg.token_dim
        else:
            cond_dim = self.cfg.kwargs.hidden_size
        return cond_dim
    
    @property
    def num_scene_tokens(self) -> int:

        return self.cfg.num_scene_tokens

    

    def _forward(
        self, 
        inputs: CompressorInputs, 
    ):
        y, pose, mask = inputs.view, inputs.pose, inputs.mask
        b, v, *_ = y.shape
        y = rearrange(y, "b v c h w -> (b v) c h w")
        pose = self.pose_embed(pose)
        pemb = None
        if self.cfg.camera_conditioning == "concat":
            pose = rearrange(pose, "b v c h w -> (b v) c h w")
            y = torch.concat([y, pose], dim=1)
        
        if self.cfg.camera_conditioning == "adaLN":
            pemb = rearrange(pose, "b v c h w -> b (v h w) c", v=v, b=b)
        
        y = self.y_embed(y)
        
        if self.cfg.camera_conditioning == "add":
            pose = rearrange(pose, "b v c h w -> (b v) (h w) c")
            y = y + pose

        y = rearrange(y, "(b v) n c -> b (v n) c", v=v, b=b)


        x = self.scene_tokens.expand(b, -1, -1)
        if self.use_scene_norm:
            x = self.scene_norm(x)
        for block in self.aggregate:
            if self.cfg.gradient_checkpointing:
                x, y, qk = checkpoint(block, x, pemb, y, self.feat_rope, self.causal, v, use_reentrant=False) 
            else:
                x, y, qk = block(x, c=pemb, y=y, feat_rope=self.feat_rope, causal=self.causal, num_views=v) 

                    
        if self.cfg.reproject_out:
            if self.norm_before_proj:
                x = self.norm(x)
            x = self.out_proj(x)
            if self.cfg.scene_token_projection == "kl":
                x = DiagonalGaussianDistribution(x)
            else:
                if not self.norm_before_proj:
                    x = self.norm(x)

        return x, None

