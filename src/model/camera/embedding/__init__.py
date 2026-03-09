

from .patch_embed import PatchEmbedding, PatchEmbeddingCfg
from .nerf_pos import NeRFPositionalEmbedding, NeRFPositionalEmbeddingCfg
from .lvsm_embed import LVSMEmbedding, LVSMEmbeddingCfg
from .time_embed import TimeEmbedding, TimeEmbeddingCfg

EmbeddingCfg = NeRFPositionalEmbeddingCfg | PatchEmbeddingCfg | LVSMEmbeddingCfg | TimeEmbeddingCfg

Embedding = PatchEmbedding | NeRFPositionalEmbedding | LVSMEmbedding | TimeEmbedding

EMBEDDING = {
    "nerf": NeRFPositionalEmbedding,
    "patch_embed": PatchEmbedding,
    "time_embed": TimeEmbedding,
    "lvsm": LVSMEmbedding 
}


def get_embedding(cfg: EmbeddingCfg, **kwargs) -> Embedding:
    
    return EMBEDDING[cfg.name](cfg, **kwargs)