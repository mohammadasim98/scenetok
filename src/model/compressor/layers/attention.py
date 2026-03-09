
import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange
from typing import Optional, Callable
from flash_attn import flash_attn_func
from timm.layers import Mlp



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
        cross_atten: bool = False,
        is_rope_3d: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        self.cross_atten = cross_atten
        self.is_rope_3d = is_rope_3d
        if use_rmsnorm:
            norm_layer = nn.RMSNorm
        
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

        
    def forward(self, x: torch.Tensor, context=None, k_rope=None, qkv_rope=None, causal=False, num_views=None, return_qk=False) -> torch.Tensor:
        B, N, C = x.shape

        if self.cross_atten:
            q = self.to_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k = self.to_k(context).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.to_v(context).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        
        q, k = self.q_norm(q), self.k_norm(k)

        if qkv_rope is not None:
            if self.is_rope_3d:
                q = qkv_rope(q)
                k = qkv_rope(k)
            else:
                q = rearrange(q, "b h (v n) d -> (b v) h n d", v=num_views)
                k = rearrange(k, "b h (v n) d -> (b v) h n d", v=num_views)
                q = qkv_rope(q)
                k = qkv_rope(k)
                q = rearrange(q, "(b v) h n d -> b h (v n) d", v=num_views)
                k = rearrange(k, "(b v) h n d -> b h (v n) d", v=num_views)
        elif self.cross_atten and k_rope is not None:
            if self.is_rope_3d:
                k = k_rope(k)
            else:
                k = rearrange(k, "b h (v n) d -> (b v) h n d", v=num_views)
                k = k_rope(k)
                k = rearrange(k, "(b v) h n d -> b h (v n) d", v=num_views)

        q = rearrange(q, "b h n d -> b n h d")
        k = rearrange(k, "b h n d -> b n h d")
        v = rearrange(v, "b h n d -> b n h d")

        x = flash_attn_func(q, k, v, dropout_p=self.attn_drop.p if self.training else 0., causal=causal)
        x = rearrange(x, "b n h d -> b n (h d)")

        x = self.proj(x)
        x = self.proj_drop(x)
        if return_qk:
            return x, (q, k)
        return x, None


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

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
        skip_last_layer: bool=True, # Useful when using DDP otherwise laster layer output is unused for which you need to use ddp_find_unused_paramters_true
        use_adaln: bool=True,
        **block_kwargs
    ):
        super().__init__()
        
        self.use_adaln = use_adaln
        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            if not skip_last_layer:
                self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = nn.RMSNorm(hidden_size)
            self.norm2 = nn.RMSNorm(hidden_size)
            if not skip_last_layer:
                self.norm3 = nn.RMSNorm(hidden_size)
            self.norm4 = nn.RMSNorm(hidden_size)
            
        # Initialize attention layer
        self.attn1 = Attention(
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
            **block_kwargs
        )

        self.attn3 = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            cross_atten=True,
            **block_kwargs
        )
        
        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            if not skip_last_layer:
                self.mlp1 = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
            self.mlp2 = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            if not skip_last_layer:
                self.mlp1 = Mlp(
                    in_features=hidden_size,
                    hidden_features=mlp_hidden_dim,
                    act_layer=approx_gelu,
                    drop=0
                )
            self.mlp2 = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )
            
        # Initialize AdaLN modulation
        if self.use_adaln:
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

        self.skip_last_layer=skip_last_layer
     

    def forward(self, x, c=None, y=None, feat_rope=None, causal=False, num_views=None, return_qk=False):
        if self.use_adaln:
            if self.wo_shift:
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=2)
                shift_msa = None
                shift_mlp = None
            else:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
            # print(x.shape, scale_msa.shape, shift_msa.shape, c.shape)
            
            y = y + gate_msa * self.attn1(modulate(self.norm1(y), shift_msa, scale_msa), context=None, qkv_rope=feat_rope, causal=False, num_views=num_views)[0]
            x = x + self.attn2(self.norm2(x), context=None, causal=causal)[0]
            
            
            attn3_out, qk = self.attn3(x, context=y, k_rope=feat_rope, causal=False, num_views=num_views, return_qk=return_qk)
            x = x + attn3_out 
            if self.skip_last_layer:
                y = None
            else:         
                y = y + gate_mlp * self.mlp1(modulate(self.norm3(y), shift_mlp, scale_mlp))
            x = x + self.mlp2(self.norm4(x))

        else:
            y = y + self.attn1(self.norm1(y), context=None, qkv_rope=feat_rope, causal=False, num_views=num_views)[0]
            x = x + self.attn2(self.norm2(x), context=None, causal=causal)[0]
            attn3_out, qk = self.attn3(x, context=y, k_rope=feat_rope, causal=False, num_views=num_views, return_qk=return_qk)
            x = x + attn3_out
            if self.skip_last_layer:
                y = None
            else:
                y = y + self.mlp1(self.norm3(y))
            
            x = x + self.mlp2(self.norm4(x))
        return x, y, qk
