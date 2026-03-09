import os
import math
import torch
import torch.nn as nn
from torch import Tensor

from pathlib import Path
from functools import reduce as re
from dataclasses import dataclass
from jaxtyping import Float, Bool
from typing import Literal, Optional, Union, Tuple
from einops import rearrange, repeat, reduce
from diffusers.models.embeddings import PatchEmbed

from .scene_generator import SceneGenerator
from .layer.lightningdit import LitDiT
from ..types import SceneGeneratorInputs
from ..camera import CameraCfg, get_camera
from ..encodings.embeddings import RotaryEmbedding1D
from ...misc.torch_utils import replace_keys_substring, pop_state_dict_by_prefix
MODEL_PREFIX = "scene_generator."

def zero_initialize(layer):
    if hasattr(layer, 'weight') and layer.weight is not None:
        nn.init.zeros_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        nn.init.zeros_(layer.bias)

@dataclass
class LightningDiTKwargsCfg:
    patch_size: int=1
    hidden_size: int=1152
    depth: int=28
    num_heads: int=16
    mlp_ratio: float=4.0
    in_channels: int=32
    learn_sigma: bool=False
    use_qknorm: bool=False
    use_swiglu: bool=True
    use_rope: bool=False
    use_rmsnorm: bool=True
    wo_shift: bool=False
    frequency_embedding_size: int=256

@dataclass
class LightningDiTCfg:


    name: Literal["lightningdit"]
    
    camera: CameraCfg
    kwargs: LightningDiTKwargsCfg
    single_dim_tokens: bool=False
    camera_conditioning: Literal["add", "adaLN"] = "adaLN"
    input_shape: Union[int, Tuple[int], list[int]]= 16
    gradient_checkpointing: bool=False
    pretrained_from: str | Path | None = None
    ckpt_path: str | Path | None = None
    load_strict: bool=True
    use_rope_1d: bool = False
    scale_factor: float = 1.0
    camera_type: Literal["avg_raymap", "flatten"] = "avg_raymap"
class LightningDiT(SceneGenerator[LightningDiTCfg]):
    def __init__(
        self, 
        cfg: LightningDiTCfg,
        cond_dim: int | None=16,
        num_scene_tokens: int=256,
        temporal_downsample: int=1,
        num_views: int=12,
    ) -> None:
        super().__init__(cfg)
        self.pretrained_from = cfg.pretrained_from
        self.num_scene_tokens = num_scene_tokens
        self.cond_dim = cond_dim
        inner_dim = cfg.kwargs.hidden_size

        self.pose_embed = get_camera(cfg.camera, embed_dim=cfg.kwargs.hidden_size, temporal_downsample=temporal_downsample)


        if self.cfg.camera_type == "avg_raymap":
            self.anchor_pose_embed = get_camera(cfg.camera, embed_dim=cfg.kwargs.hidden_size, temporal_downsample=temporal_downsample)

        elif self.cfg.camera_type == "flatten":
            self.anchor_pose_embed = nn.Sequential(
                nn.Linear(25, cfg.kwargs.hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(cfg.kwargs.hidden_size, cfg.kwargs.hidden_size, bias=True),
            )
        print("(Scene Generator) Using gradient checkpointing: ", cfg.gradient_checkpointing)
        print("(Scene Generator) Using camera type: ", cfg.camera_type)

        self.model = LitDiT(
            input_size=cfg.input_shape,
            patch_size=cfg.kwargs.patch_size,
            cond_dim=cond_dim,
            in_channels=cfg.kwargs.in_channels,
            hidden_size=cfg.kwargs.hidden_size,
            depth=cfg.kwargs.depth,
            num_heads=cfg.kwargs.num_heads,
            mlp_ratio=cfg.kwargs.mlp_ratio,
            learn_sigma=cfg.kwargs.learn_sigma,
            use_qknorm=cfg.kwargs.use_qknorm,
            use_swiglu=cfg.kwargs.use_swiglu,
            use_rmsnorm=cfg.kwargs.use_rmsnorm,
            wo_shift=cfg.kwargs.wo_shift,
            use_checkpoint=cfg.gradient_checkpointing,
            frequency_embedding_size=cfg.kwargs.frequency_embedding_size,
            num_tokens=num_scene_tokens,
            num_views=num_views
        )
        if self.cfg.use_rope_1d:
            half_head_dim = cfg.kwargs.hidden_size // cfg.kwargs.num_heads
            self.anchor_rope = RotaryEmbedding1D(half_head_dim, seq_len=num_views)
        else:
            self.anchor_rope = None
        if self.pretrained_from is not None:
            print("(Denoiser) Loading from pretrained: ", self.pretrained_from)
            weights = torch.load(self.pretrained_from, map_location=torch.device("cpu"))["model"]
            weights = pop_state_dict_by_prefix(weights, "module.")

            self.model.load_state_dict(weights, strict=False)

        # if cond_dim is not None:
        #     self.cnd_proj = nn.Linear(cond_dim, inner_dim)



        # if self.cnd_proj is not None:
        #     print("(Denoiser) Zero initializing scene token projection")
        #     self.cnd_proj.apply(zero_initialize)
        

        if cfg.ckpt_path is not None:
            self.load_weights(cfg.ckpt_path, strict=cfg.load_strict)

    def load_weights(
        self,
        path: Path | str,
        **kwargs
    ):  
        print(f"(Denoiser) Loading weights (strict={self.cfg.load_strict}) from: ", path)
        weights = torch.load(path)
        weights = pop_state_dict_by_prefix(weights["state_dict"], MODEL_PREFIX)
        if not self.cfg.load_strict:
            for key in list(weights.keys()):
                try:
                    param_shape = re(getattr, [self, *key.split(".")]).shape
                    if param_shape != weights[key].shape:
                        del weights[key]
                except AttributeError:
                    continue
        self.load_state_dict(weights, **kwargs)

    def _forward(
        self,
        inputs: SceneGeneratorInputs,
        cond_mask: Bool[Tensor, "batch cond_view"]
    ) -> Float[Tensor, "batch view channel height width"]:
        

        cond_state, pose, anchor_pose, timestep, state = inputs.view, inputs.pose, inputs.anchor_pose, inputs.timestep, inputs.state
        # latents = latents.to(torch.half)
        pemb = self.pose_embed(pose, temporal_downsample=1)

        if self.cfg.camera_type == "avg_raymap":
            anchor_pemb = self.anchor_pose_embed(anchor_pose, temporal_downsample=1)
            anchor_pemb = reduce(anchor_pemb, "b v c h w -> b v c", reduction="mean")
        elif self.cfg.camera_type == "flatten":
            anchor_pose = anchor_pose.flatten()
            anchor_pemb = self.anchor_pose_embed(anchor_pose)
        else:
            raise NotImplementedError(f"Incorrect value for camera type: {self.cfg.camera_type}")    
        pemb = rearrange(pemb, "b v c h w -> b v (h w) c")
        sample, qk_list = self.model(state=state, pose=pemb.bfloat16(), timestep=timestep, cond_state=cond_state, anchor_pose=anchor_pemb.bfloat16(), cond_mask=cond_mask)
        return sample, qk_list
