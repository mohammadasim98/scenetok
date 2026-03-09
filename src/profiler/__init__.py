
from lightning.pytorch.profilers import PyTorchProfiler, Profiler, SimpleProfiler

from .pytorch import PyTorchProfilerCfg
from .simple import SimpleProfilerCfg
from .base import BaseProfilerCfg
from dataclasses import asdict

PROFILER: dict[str, Profiler] = {
    "base": Profiler,
    "simple": SimpleProfiler,
    "pytorch": PyTorchProfiler
}

ProfilerCfg = None | BaseProfilerCfg | SimpleProfilerCfg | PyTorchProfilerCfg


def get_profiler(
    cfg: ProfilerCfg,
) -> Profiler:
    
    if cfg == None:
        return None
    _dict = asdict(cfg)
    del _dict["name"]
    return PROFILER[cfg.name](**_dict)