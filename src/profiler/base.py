
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union, Literal

@dataclass
class BaseProfilerCfg:
    name: Literal["base"]
    dirpath: Optional[Union[str, Path]] = None
    filename: Optional[str] = None