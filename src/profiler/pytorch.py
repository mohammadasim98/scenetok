



from dataclasses import dataclass
from typing import Optional, Any, Literal, Callable, Iterable
from torch.profiler.profiler import _ITraceObserver, _ExperimentalConfig, ProfilerAction, ProfilerActivity

from .base import BaseProfilerCfg

@dataclass
class PyTorchProfilerCfg(BaseProfilerCfg):
    name: Literal["pytorch"]
    group_by_input_shapes: bool = False
    emit_nvtx: bool = False
    export_to_chrome: bool = True
    row_limit: int = 20
    sort_by_key: Optional[str] = None
    record_module_names: bool = True
    table_kwargs: Optional[dict[str, Any]] = None
    
    # From torch
    activities: Optional[Iterable[ProfilerActivity]] = None
    schedule: Optional[Callable[[int], ProfilerAction]] = None
    on_trace_ready: Optional[Callable[..., Any]] = None
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    with_flops: bool = False
    with_modules: bool = False
    experimental_config: Optional[_ExperimentalConfig] = None
    execution_trace_observer: Optional[_ITraceObserver] = None
    acc_events: bool = False