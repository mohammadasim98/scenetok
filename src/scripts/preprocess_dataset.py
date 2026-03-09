import hydra
from pathlib import Path
from dataclasses import dataclass
from omegaconf import DictConfig
from tqdm import tqdm
from src.config import load_typed_config
from src.global_cfg import set_cfg
from src.scripts.precompute_latents import get_latent_dataset, LatentDatasetCfg

@dataclass
class PreprocessCfg:

    dataset: LatentDatasetCfg
    stage: str
    index: int = 0
    size: int = None
    output_dir: Path = Path("")
    flip: bool = False


@hydra.main(
    version_base=None,
    config_path="../../config/scripts",
    config_name="preprocess_config",
)
def preprocess(cfg_dict: DictConfig):
    cfg = load_typed_config(cfg_dict, PreprocessCfg)
    
    set_cfg(cfg_dict)

    dataset = get_latent_dataset(
        cfg=cfg.dataset, 
        stage=cfg.stage, 
        index=cfg.index, 
        size=cfg.size, 
        output_dir=cfg.output_dir, 
        flip=cfg.flip
    )

    for i in tqdm(range(len(dataset)),desc=f"[Index: {cfg.index}] Preprocessing {cfg.size} chunks"):
        dataset[i]

    exit(0)

if __name__ == "__main__":

    preprocess()

