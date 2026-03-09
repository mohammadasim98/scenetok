
import torch

from .lightningdit import LightningDiT, LightningDiTCfg


SceneGeneratorCfg = LightningDiTCfg 
SceneGenerator = LightningDiT 

SCENE_GENERATOR = {
    "lightningdit": LightningDiT,

}


def get_scene_generator(
    scene_generator_cfg: SceneGeneratorCfg,
    **kwargs
) -> SceneGenerator:
    return SCENE_GENERATOR[scene_generator_cfg.name](scene_generator_cfg, **kwargs)


