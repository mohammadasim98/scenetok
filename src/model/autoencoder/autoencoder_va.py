from typing import Literal, Optional, Tuple, List
from dataclasses import dataclass
from typing import Union
from pathlib import Path
# Refer to https://github.com/huggingface/diffusers/blob/6131a93b969f87d171148bd367fd9990d5a49b6b/src/diffusers/models/autoencoders/autoencoder_kl.py#L38

@dataclass
class VAKwargsCfg:
    latent_channels: int=32
    scaling_factor: float=1.0

@dataclass
class AutoencoderVACfg:
    name: Literal["va"]
    pretrained_from: str
    kwargs: VAKwargsCfg