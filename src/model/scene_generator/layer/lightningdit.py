"""
Lightning DiT's codes are built from original DiT & SiT.
(https://github.com/facebookresearch/DiT; https://github.com/willisma/SiT)
It demonstrates that a advanced DiT together with advanced diffusion skills
could also achieve a very promising result with 1.35 FID on ImageNet 256 generation.

Enjoy everyone, DiT strikes back!

by Maple (Jingfeng Yao) from HUST-VL
"""

import os
import math
import numpy as np
from jaxtyping import Bool
from torch import Tensor
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed, Mlp
from .swiglu_ffn import SwiGLUFFN 
from .pos_embed import VisionRotaryEmbeddingFast
from .rmsnorm import RMSNorm
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

def modulate(x, shift, scale):
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift

class Attention(nn.Module):
    """
    Attention module of LightningDiT.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
        use_rmsnorm: bool = False,
        cross_atten: bool = False
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        self.cross_atten = cross_atten
        if use_rmsnorm:
            norm_layer = RMSNorm
        
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        if not cross_atten:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        else:
            self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
            self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
            self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        
    def forward(self, x: torch.Tensor, context=None, rope=None) -> torch.Tensor:
        B, N, C = x.shape

        if self.cross_atten:
            q = self.to_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.to_k(context).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.to_v(context).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            # q = rope(q)
            # if not self.cross_atten:
            print("Using 1D ROPE")
            k[:, -rope.length:] = rope(k[:, -rope.length:])
        q = rearrange(q, "b h n d -> b n h d")
        k = rearrange(k, "b h n d -> b n h d")
        v = rearrange(v, "b h n d -> b n h d")

        x = flash_attn_func(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        x = rearrange(x, "b n h d -> b n (h d)")

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Same as DiT.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        Args:
            t: A 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim: The dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.
        Returns:
            An (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[..., None].float() * freqs[None, None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding
    
    def forward(self, t: torch.Tensor, pemb: torch.Tensor | None = None) -> torch.Tensor:

        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if pemb is not None:

            return self.mlp(t_freq) + pemb

        return self.mlp(t_freq)




class LightningDiTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including: 
    - ROPE
    - QKNorm 
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False, 
        use_rmsnorm=False,
        wo_shift=False,
        **block_kwargs
    ):
        super().__init__()
        
        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            
        # Initialize attention layer
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )

        self.attn2 = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            cross_atten=True,
            **block_kwargs
        )

        # self.attn3 = Attention(
        #     hidden_size,
        #     num_heads=num_heads,
        #     qkv_bias=True,
        #     qk_norm=use_qknorm,
        #     use_rmsnorm=use_rmsnorm,
        #     cross_atten=True,
        #     **block_kwargs
        # )
        
        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )
            
        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    def forward(self, x, c, y, feat_rope=None):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=2)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
        

        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), context=None)
        x = x + self.attn2(x, context=y, rope=feat_rope)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of LightningDiT.
    """
    def __init__(self, hidden_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LitDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        cond_dim=64,
        in_channels=64,
        hidden_size=1152,
        num_tokens=512,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
        use_qknorm=False,
        use_swiglu=False,
        use_rmsnorm=False,
        wo_shift=False,
        use_checkpoint=False,
        frequency_embedding_size: int=256,
        num_views: int=12
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels if not learn_sigma else in_channels * 2
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.use_checkpoint = use_checkpoint
        self.num_tokens = num_tokens
        max_shape = max(input_size)
        self.x_embedder = nn.Linear(cond_dim, hidden_size, bias=True)
        self.c_embedder = PatchEmbed(max_shape, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size=frequency_embedding_size)
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_size), requires_grad=True)


        self.blocks = nn.ModuleList([
            LightningDiTBlock(hidden_size, 
                     num_heads, 
                     mlp_ratio=mlp_ratio, 
                     use_qknorm=use_qknorm, 
                     use_swiglu=use_swiglu, 
                     use_rmsnorm=use_rmsnorm,
                     wo_shift=wo_shift,
                     ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, out_channels=cond_dim, use_rmsnorm=use_rmsnorm)
        self.null_tokens = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], self.num_tokens)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.c_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.c_embedder.proj.bias, 0)


        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(
        self, 
        state: torch.Tensor,
        pose: torch.Tensor = None,
        anchor_pose: torch.Tensor = None,
        timestep: torch.Tensor = None,
        cond_state: torch.Tensor = None,
        cond_mask: Optional[Bool[Tensor, "batch cond_view"]] = None,
        anchor_rope = None
    ):
        """
        Forward pass of LightningDiT.
        x: (B, N, C) tensor of spatial inputs (images or latent representations of images)
        t: (B,) tensor of diffusion timesteps
        y: (B,) tensor of class labels
        use_checkpoint: boolean to toggle checkpointing
        """
        x, t, p, ap, y = state, timestep, pose, anchor_pose, cond_state
        use_checkpoint = self.use_checkpoint
        b, v, _, h, w = y.shape


        x = self.x_embedder(x) + self.pos_embed

        y = rearrange(y, "b v c h w -> (b v) c h w")
        y = self.c_embedder(y)  # (N, T, D), where T = H * W / patch_size ** 2
        y = rearrange(y, "(b v) n d -> b v n d", v=v)
        
        # t = rearrange(t, "b v -> (b v)")
        t = self.t_embedder(t, pemb=None)  
        # t = rearrange(t, "b d -> b () d")

        y = y + p
        if cond_mask is not None:
            null_tokens =  self.null_tokens.to(y.dtype)
            y = y * cond_mask[..., None, None].to(y.dtype) + null_tokens[..., None, :] * (~cond_mask[..., None, None]).to(y.dtype)
        y = rearrange(y , "b v n c -> b (v n) c", v=v)


        y = torch.concat([y, ap], dim=1)

        for block in self.blocks:
            # print(x.dtype, c.dtype, y.dtype)
            if use_checkpoint:
                x = checkpoint(block, x, t, y, anchor_rope, use_reentrant=True)
            else:
                x = block(x, t, y=y, feat_rope=anchor_rope)

        x = self.final_layer(x, t)             


        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x, None

    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=None, cfg_interval_start=None):
        """
        Forward pass of LightningDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        
        if cfg_interval is True:
            timestep = t[0]
            if timestep < cfg_interval_start:
                half_eps = cond_eps

        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# #################################################################################
# #                             LightningDiT Configs                              #
# #################################################################################

# def LightningDiT_XL_1(**kwargs):
#     return LightningDiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)

# def LightningDiT_XL_2(**kwargs):
#     return LightningDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

# def LightningDiT_L_2(**kwargs):
#     return LightningDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

# def LightningDiT_B_1(**kwargs):
#     return LightningDiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

# def LightningDiT_B_2(**kwargs):
#     return LightningDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

# def LightningDiT_1p0B_1(**kwargs):
#     return LightningDiT(depth=24, hidden_size=1536, patch_size=1, num_heads=24, **kwargs)

# def LightningDiT_1p0B_2(**kwargs):
#     return LightningDiT(depth=24, hidden_size=1536, patch_size=2, num_heads=24, **kwargs)

# def LightningDiT_1p6B_1(**kwargs):
#     return LightningDiT(depth=28, hidden_size=1792, patch_size=1, num_heads=28, **kwargs)

# def LightningDiT_1p6B_2(**kwargs):
#     return LightningDiT(depth=28, hidden_size=1792, patch_size=2, num_heads=28, **kwargs)

# LightningDiT_models = {
#     'LightningDiT-B/1': LightningDiT_B_1, 'LightningDiT-B/2': LightningDiT_B_2,
#     'LightningDiT-L/2': LightningDiT_L_2,
#     'LightningDiT-XL/1': LightningDiT_XL_1, 'LightningDiT-XL/2': LightningDiT_XL_2,
#     'LightningDiT-1p0B/1': LightningDiT_1p0B_1, 'LightningDiT-1p0B/2': LightningDiT_1p0B_2,
#     'LightningDiT-1p6B/1': LightningDiT_1p6B_1, 'LightningDiT-1p6B/2': LightningDiT_1p6B_2,
# }