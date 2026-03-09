
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union, Literal

from .base import BaseProfilerCfg


@dataclass
class SimpleProfilerCfg(BaseProfilerCfg):
    name: Literal["simple"]
    extended: bool = True,