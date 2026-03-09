
import torch

from pathlib import Path
from jaxtyping import Float
from torch import nn, Tensor
from dataclasses import dataclass, field
from typing import Optional, Union, Literal
from einops import rearrange

from .camera import Camera
from .embedding import get_embedding, EmbeddingCfg
from ..types import CameraInputs
from ...misc.camera_utils import generate_image_rays

@dataclass
class PluckerCfg:
    name: Literal["plucker"]
    input_shape: Union[list[int, int], tuple[int, int]]
    scale: Union[list[float, float], tuple[float, float]]=field(default_factory=[1.0, 1.0])
    normalize: bool = True
    origin_first: bool = False
    embedding: Optional[EmbeddingCfg] = None

class Plucker(Camera[PluckerCfg]):
    cfg: PluckerCfg
    model: Optional[nn.Module]

    def __init__(
        self, 
        cfg:PluckerCfg,
        num_split: int=1,
        using_wan: bool=False,
        **kwargs
    ) -> None:
        super().__init__(cfg)
        self.using_wan = using_wan
        self.num_split = num_split
        print(f"(Plucker) Using Wan latents: {self.using_wan}")

        if cfg.embedding is not None:
            self.model = get_embedding(cfg.embedding, input_shape=cfg.input_shape, **kwargs)
        else:
            self.model = nn.Identity()

    def load_weights(
        self,
        path: Path | str,
        **kwargs
    ):  
        if self.model is None or path is None:
            return
        weights = torch.load(path)
        self.model.load_state_dict(weights, **kwargs)

    def forward(
        self,
        inputs: CameraInputs,
        temporal_downsample: int=1
    ) -> Float[Tensor, "batch ... channel _ _"]:
        
        intrinsics, extrinsics = inputs.intrinsics, inputs.extrinsics
        intrinsics[..., 0, 0] *= self.cfg.scale[0]
        intrinsics[..., 1, 1] *= self.cfg.scale[1]
        orig_dtype = extrinsics.dtype


        _, origins, directions = generate_image_rays(self.cfg.input_shape, extrinsics.float(), intrinsics.float(), self.cfg.normalize)

        origins = torch.cross(origins, directions, dim=2)

        if self.cfg.origin_first: # backward compatibility for legacy experiments
            ray_encodings = torch.concat([origins, directions], dim=2)   
        else:  
            ray_encodings = torch.concat([directions, origins], dim=2) 

        if self.using_wan:

            # First video token encodes one image
            # Need to pad it 
            n = ray_encodings.shape[1] // 17
            ray_encodings = torch.chunk(ray_encodings, n, dim=1)
            _list = []
            for ray_encoding in ray_encodings:
                padding = torch.zeros_like(ray_encoding[:, 0:temporal_downsample-1], device=ray_encoding.device, dtype=ray_encoding.dtype)

                ray_encoding = torch.concat(
                    [ray_encoding[:, 0:1], 
                    padding,
                    ray_encoding[:, 1:]
                    ], dim=1) 
                _list.append(ray_encoding)
            ray_encodings = torch.concat(_list, dim=1)


        if temporal_downsample > 1:
            ray_encodings = rearrange(ray_encodings, "b (v t) c h w -> b v (t c) h w", t=temporal_downsample)
        

        return self.model(ray_encodings).to(orig_dtype)
