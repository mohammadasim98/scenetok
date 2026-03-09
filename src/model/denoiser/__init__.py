

from .lightningdit import LightningDiT, LightningDiTCfg


DenoiserCfg = LightningDiTCfg
Denoiser = LightningDiT

DENOISER = {
    "lightningdit": LightningDiT,
}


def get_denoiser(
    denoiser_cfg: DenoiserCfg,
    **kwargs
) -> Denoiser:
    return DENOISER[denoiser_cfg.name](denoiser_cfg, **kwargs)


