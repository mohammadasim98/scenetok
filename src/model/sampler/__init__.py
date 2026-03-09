import torch


from .autoregressive import AutoRegressiveSampler, AutoRegressiveSamplerCfg
from .pyramid import PyramidSampler, PyramidSamplerCfg
from .full_sequence import FullSequenceSampler, FullSequenceSamplerCfg


SamplerCfg = AutoRegressiveSamplerCfg | PyramidSamplerCfg | FullSequenceSamplerCfg
Sampler = AutoRegressiveSampler | PyramidSampler | FullSequenceSampler

SAMPLER: dict = {
    "autoregressive": AutoRegressiveSampler,
    "pyramid": PyramidSampler,
    "full_sequence": FullSequenceSampler
}

SamplerListCfg = list[SamplerCfg]
SamplerList = list[Sampler]

def get_samplers(
    cfg: SamplerListCfg
) -> SamplerList:
    
    return [SAMPLER[c.name](c) for c in cfg]
        
def get_sampler(
    cfg: SamplerCfg
) -> Sampler:
    
    return SAMPLER[cfg.name](cfg)