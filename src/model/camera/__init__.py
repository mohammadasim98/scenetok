
from .plucker import Plucker, PluckerCfg
from .ray import Ray, RayCfg

from dataclasses import dataclass
from jaxtyping import Float
from torch import Tensor


CameraCfg = PluckerCfg | RayCfg
Camera = Plucker | Ray

CAMERA = {
    "plucker": Plucker,
    "ray": Ray
}

def get_camera(cfg: CameraCfg, **kwargs) -> Camera:

    return CAMERA[cfg.name](cfg, **kwargs)