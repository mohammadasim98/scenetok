

from .mvae_compressor import MVAECompressor, MVAECompressorCfg


COMPRESSOR = {
    "mvae_compressor": MVAECompressor,
}

Compressor = MVAECompressor
CompressorCfg = MVAECompressorCfg 


def get_compressor(
    cfg: CompressorCfg,
    **kwargs
) -> Compressor:

    return COMPRESSOR[cfg.name](cfg, **kwargs)