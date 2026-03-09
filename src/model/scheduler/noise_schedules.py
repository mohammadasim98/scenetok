import torch
import math
def cosine_simple_diffusion_schedule(
    timesteps,
    logsnr_min=-15.0,
    logsnr_max=15.0,
    shifted: float = 1.0,
    interpolated: bool = False,
):
    """
    cosine schedule with different parameterization
    following Simple Diffusion - https://arxiv.org/abs/2301.11093
    Supports "shifted cosine schedule" and "interpolated cosine schedule"

    Args:
        timesteps: number of timesteps
        logsnr_min: minimum log SNR
        logsnr_max: maximum log SNR
        shifted: shift the schedule by a factor. Should be base_resolution / current_resolution
        interpolated: interpolate between the original and the shifted schedule, requires shifted != 1.0
    """
    t_min = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max, dtype=torch.float64)))
    t_max = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min, dtype=torch.float64)))
    t = torch.linspace(0, 1, timesteps, dtype=torch.float64)
    logsnr = -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))
    if shifted != 1.0:
        shifted_logsnr = logsnr + 2 * torch.log(
            torch.tensor(shifted, dtype=torch.float64)
        )
        if interpolated:
            logsnr = t * logsnr + (1 - t) * shifted_logsnr
        else:
            logsnr = shifted_logsnr

    alphas_cumprod = 1 / (1 + torch.exp(-logsnr))
    return alphas_cumprod