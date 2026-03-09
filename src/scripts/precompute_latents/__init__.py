

from src.scripts.precompute_latents.va_re10k import LatentVARE10KDataset, LatentVARE10KDatasetCfg
from src.scripts.precompute_latents.va_dl3dv import LatentVADL3DVDataset, LatentVADL3DVDatasetCfg
from src.scripts.precompute_latents.videodc_re10k import LatentVideoDCRE10KDataset, LatentVideoDCRE10KDatasetCfg
from src.scripts.precompute_latents.videodc_dl3dv import LatentVideoDCDL3DVDataset, LatentVideoDCDL3DVDatasetCfg



LatentDatasetCfg = (
    LatentVARE10KDatasetCfg
    | LatentVADL3DVDatasetCfg
    | LatentVideoDCRE10KDatasetCfg
    | LatentVideoDCDL3DVDatasetCfg
)
LatentDataset = (
    LatentVARE10KDataset 
    | LatentVADL3DVDataset
    | LatentVideoDCRE10KDataset
    | LatentVideoDCDL3DVDataset
)

DATASET: dict[str, LatentDataset] = {
    "va_re10k": LatentVARE10KDataset,
    "videodc_re10k": LatentVideoDCRE10KDataset,
    "va_dl3dv": LatentVADL3DVDataset,
    "videodc_dl3dv": LatentVideoDCDL3DVDataset,

}


def get_latent_dataset(
    cfg: LatentDatasetCfg,
    **kwargs
) -> LatentDataset:
    return DATASET[cfg.name](cfg, **kwargs)