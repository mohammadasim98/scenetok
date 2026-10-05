

from torch import Generator
from torch.utils.data import Dataset

# Custom modules
from .dtypes import Stage
from .view_sampler import get_view_sampler
from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .dataset_dl3dv import DatasetDL3DV, DatasetDL3DVCfg
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .dataset_latent import DatasetLatent, DatasetLatentCfg
from .dataset_re10k_hires import DatasetRE10kHiRes, DatasetRE10kHiResCfg


DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
    "dl3dv": DatasetDL3DV,
    "latent": DatasetLatent,
    "re10k_hires": DatasetRE10kHiRes,
}


DatasetCfg = (
    DatasetDL3DVCfg 
    | DatasetLatentCfg
    | DatasetRE10kCfg
    | DatasetRE10kHiResCfg
)


def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
    step_tracker: StepTracker | None,
    generator: Generator | None = None,
    force_shuffle: bool = False
) -> Dataset:

    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.cameras_are_circular,
        step_tracker,
        generator
    )
    return DATASETS[cfg.name](cfg, stage, view_sampler, force_shuffle)
