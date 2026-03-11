
import torch
import torch.nn as nn
from torch import Tensor

from pathlib import Path
from functools import reduce 
from dataclasses import dataclass
from jaxtyping import Float
from typing import Literal,Union, Tuple
from einops import rearrange

from .denoiser import Denoiser
from .layers.lightningdit import LitDiT
from ..types import DenoiserInputs
from ..camera import CameraCfg, get_camera
from ...misc.torch_utils import pop_state_dict_by_prefix

MODEL_PREFIX = "denoiser."

def zero_initialize(layer):
    if hasattr(layer, 'weight') and layer.weight is not None:
        nn.init.zeros_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        nn.init.zeros_(layer.bias)

@dataclass
class LightningDiTKwargsCfg:
    patch_size: int=1
    in_channels: int=32
    hidden_size: int=1152
    depth: int=28
    num_heads: int=16
    mlp_ratio: float=4.0
    class_dropout_prob: float=0.1
    num_classes: int=1000
    learn_sigma: bool=False
    use_qknorm: bool=False
    use_swiglu: bool=True
    use_rope: bool=True
    use_rope_3d: bool=True
    use_rmsnorm: bool=True
    wo_shift: bool=False
    frequency_embedding_size: int=256

@dataclass
class LightningDiTCfg:


    name: Literal["lightningdit"]
    
    camera: CameraCfg
    kwargs: LightningDiTKwargsCfg
    single_dim_tokens: bool=False
    num_target_split: int=1
    camera_conditioning: Literal["add", "adaLN"] = "adaLN"
    input_shape: Union[int, Tuple[int], list[int]]= 16
    gradient_checkpointing: bool=False
    pretrained_from: str | Path | None = None
    ckpt_path: str | Path | None = None
    load_strict: bool=True
    causal_attention: bool=False
class LightningDiT(Denoiser[LightningDiTCfg]):
    def __init__(
        self, 
        cfg: LightningDiTCfg,
        cond_dim: int | None=16,
        num_scene_tokens: int=256,
        num_views: int=8,
        temporal_downsample: int=1,
        using_wan: bool=False,
        cfg_train: bool=False,
        **kwargs
    ) -> None:
        super().__init__(cfg)
        self.pretrained_from = cfg.pretrained_from
        self.num_scene_tokens = num_scene_tokens
        self.cond_dim = cond_dim
        inner_dim = cfg.kwargs.hidden_size
        num_split = cfg.num_target_split
        self.pose_embed = get_camera(cfg.camera, num_split=num_split, using_wan=using_wan, embed_dim=cfg.kwargs.hidden_size, temporal_downsample=temporal_downsample)
        print("(Denoiser) Using gradient checkpointing: ", cfg.gradient_checkpointing)

        self.model = LitDiT(
            input_size=cfg.input_shape,
            patch_size=cfg.kwargs.patch_size,
            in_channels=cfg.kwargs.in_channels,
            hidden_size=cfg.kwargs.hidden_size,
            depth=cfg.kwargs.depth,
            num_heads=cfg.kwargs.num_heads,
            mlp_ratio=cfg.kwargs.mlp_ratio,
            class_dropout_prob=cfg.kwargs.class_dropout_prob,
            num_classes=cfg.kwargs.num_classes,
            learn_sigma=cfg.kwargs.learn_sigma,
            use_qknorm=cfg.kwargs.use_qknorm,
            use_swiglu=cfg.kwargs.use_swiglu,
            use_rope=cfg.kwargs.use_rope,
            use_rope_3d=cfg.kwargs.use_rope_3d,
            use_rmsnorm=cfg.kwargs.use_rmsnorm,
            wo_shift=cfg.kwargs.wo_shift,
            use_checkpoint=cfg.gradient_checkpointing,
            frequency_embedding_size=cfg.kwargs.frequency_embedding_size,
            num_views=num_views,
            num_split=num_split,
            causal_attention=cfg.causal_attention
        )
        
        if self.pretrained_from is not None:
            print("(Denoiser) Loading from pretrained: ", self.pretrained_from)
            weights = torch.load(self.pretrained_from, map_location=torch.device("cpu"))["model"]
            weights = pop_state_dict_by_prefix(weights, "module.")

            self.model.load_state_dict(weights, strict=False)

        self.cnd_proj = nn.Linear(cond_dim, inner_dim)

        if cfg_train:
            self.null_tokens = nn.Parameter(torch.zeros(1, 1, inner_dim))

        if cfg.ckpt_path is not None:
            self.load_weights(cfg.ckpt_path, strict=cfg.load_strict)

    def load_weights(
        self,
        path: Path | str,
        **kwargs
    ):  
        print(f"(Denoiser) Loading weights (strict={self.cfg.load_strict}) from: ", path)
        weights = torch.load(path, map_location=torch.device("cpu"))
        weights = pop_state_dict_by_prefix(weights["state_dict"], MODEL_PREFIX)
        if not self.cfg.load_strict:
            for key in list(weights.keys()):
                try:
                    param_shape = reduce(getattr, [self, *key.split(".")]).shape
                    if param_shape != weights[key].shape:
                        del weights[key]
                except AttributeError:
                    continue
        self.load_state_dict(weights, **kwargs)

    def _forward(
        self,
        inputs: DenoiserInputs,
        temporal_downsample: int
    ) -> Float[Tensor, "batch view channel height width"]:
        

        latents, pose, timestep, state = inputs.view, inputs.pose, inputs.timestep, inputs.state
        # latents = latents.to(torch.half)
        pemb = self.pose_embed(pose, temporal_downsample=temporal_downsample)

        pemb = rearrange(pemb, "b v c h w -> b v (h w) c")
        sample, qk_list = self.model(latents=latents, pose=pemb.bfloat16(), timestep=timestep, cond_state=state)
        return sample, qk_list
