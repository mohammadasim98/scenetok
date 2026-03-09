import numpy as np
from dataclasses import dataclass, asdict
from typing import Literal, Union
from diffusers import DDIMScheduler, DDPMScheduler

from .ddim import DDIMSchedulerCfg
from .ddpm import DDPMSchedulerCfg
from .flow import RectifiedFlowMatchingSchedulerCfg, RectifiedFlowMatchingScheduler

SchedulerCfg = DDIMSchedulerCfg | DDPMScheduler | RectifiedFlowMatchingSchedulerCfg
Scheduler = DDIMScheduler | DDPMSchedulerCfg | RectifiedFlowMatchingScheduler


SCHEDULER = {
    "ddim": DDIMScheduler,
    "ddpm": DDPMScheduler,
    "rectified_flow": RectifiedFlowMatchingScheduler
}

def _parse_cfg(cfg):

    # Original type and types returned by hydra may conflict
    try:
        if type(cfg.trained_betas) == list:
            cfg.trained_betas = np.array(cfg.trained_betas)
    except AttributeError as err:
        print(err)

    return cfg

def get_scheduler(
    cfg: SchedulerCfg
) -> Union[DDPMScheduler, DDIMScheduler, RectifiedFlowMatchingScheduler]:
     
    if cfg.pretrained_from is None:
        return SCHEDULER[cfg.name](**asdict(_parse_cfg(cfg.kwargs)))

    else:
        return SCHEDULER[cfg.name].from_pretrained(cfg.pretrained_from, subfolder="scheduler")
