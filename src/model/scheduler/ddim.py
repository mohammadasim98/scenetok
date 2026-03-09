

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional


# Refer to https://github.com/huggingface/diffusers/blob/91ddd2a25b848df0fa1262d4f1cd98c7ccb87750/src/diffusers/schedulers/scheduling_ddim.py#L77
@dataclass
class DDIMKwargsCfg:
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    trained_betas: Optional[np.ndarray | list] = None
    clip_sample: bool = True
    set_alpha_to_one: bool = True
    steps_offset: int = 0
    rescale_betas_zero_snr: bool = False    
    prediction_type: Literal["epsilon", "sample", "v_prediction"] = "epsilon"


@dataclass
class DDIMSchedulerCfg:
    name: Literal["ddim"]
    num_train_timesteps: int
    num_inference_steps: int
    pretrained_from: str | None
    sampling_type: Literal["random_uniform", "random_independent"]
    kwargs: DDIMKwargsCfg