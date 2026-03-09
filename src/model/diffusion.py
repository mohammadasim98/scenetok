import torch

from tqdm import tqdm
from torch import Tensor
from einops import repeat, rearrange
from typing import Optional
from jaxtyping import Float
from src.model.sampler import Sampler
from src.model.denoiser import Denoiser
from src.model.types import CameraInputs, DenoiserInputs
from src.model.scheduler import SchedulerCfg, Scheduler
from src.model.autoencoder import Autoencoder

# VA-VAE latent statistics
latent_mean = torch.tensor([ 0.5985, -0.4992,  0.6440, -0.0971, -1.1910, -1.4332,  0.4685,  0.6259,
         0.6320, -0.4897, -0.7445,  1.1596,  0.8456,  0.5008,  0.2293,  0.4754,
        -0.4379,  0.8317, -0.0751,  0.3063,  0.4665, -0.0914, -0.8271,  0.0781,
         1.4151,  1.3792,  0.2696, -0.7573,  0.2813, -0.3092,  0.0779,  0.3497])[None, None, :, None, None]
latent_std = torch.tensor([3.8461, 4.2699, 3.5768, 3.5911, 3.6231, 3.4810, 3.3075, 3.5093, 3.5541,
        3.6067, 3.7058, 3.6314, 3.6295, 3.6205, 3.2590, 3.1868, 3.8258, 3.5999,
        3.2966, 3.2261, 3.2192, 3.1055, 3.5805, 4.3569, 3.3085, 3.2076, 4.5150,
        3.4870, 3.0416, 3.4869, 4.4310, 4.0881])[None, None, :, None, None]

@torch.no_grad()
def first_stage_encode(
    autoencoder: Autoencoder, 
    inputs: Float[Tensor, "b v c h w"], 
    view_type: str, 
    autoencoder_name: Optional[str]=None,
    scaling_factor: float=1.0
):
    b, v, c, h, w = inputs.shape
    inputs = rearrange(inputs, "b v c h w -> (b v) c h w ")
    inputs = inputs * 2.0 - 1.0 
    if autoencoder_name is not None:
        
        if autoencoder_name == "dc":
            with torch.autocast(enabled=False, device_type="cuda", dtype=torch.bfloat16):
                latents = autoencoder[view_type].encode(inputs).latent
                latents = rearrange(latents, "(b v) c h w -> b v c h w", v=v)
        elif autoencoder_name == "video_dc":
            with torch.autocast(enabled=False, device_type="cuda", dtype=torch.bfloat16):
                inputs = rearrange(inputs, "(b v) c h w -> b v c h w", v=v)
                latents = autoencoder[view_type].encode(inputs)
        elif autoencoder_name == "wan":
            inputs = rearrange(inputs, "(b v) c h w -> b v c h w", v=v)
            latents = autoencoder[view_type].encode(inputs)
        elif autoencoder_name == "wan_single":
            inputs = rearrange(inputs, "(b v) c h w -> b v c h w", v=v)
            latents = []
            for i in range(v):
                inputs_single = inputs[:, i:i+1, :, :, :]
                latents.append(autoencoder[view_type].encode(inputs_single))
            latents = torch.concat(latents, dim=1)
            latents = rearrange(inputs, "(b v) c h w -> b v c h w", b=b)
        else:
            latents = autoencoder[view_type].encode(inputs).latent_dist.sample()
            latents = rearrange(latents, "(b v) c h w -> b v c h w", v=v)
        
        if autoencoder_name == "va":
            latents = (latents - latent_mean.to(latents[0].device)) / latent_std[0].to(latents.device)
        else:
            latents = latents * scaling_factor
    else:
        latents = rearrange(inputs, "(b v) c h w -> b v c h w", b=b)

    return latents

@torch.no_grad()
def last_stage_decode(
    autoencoder: Autoencoder, 
    latents: Float[Tensor, "b v c h w"], 
    view_type: str, 
    autoencoder_name: Optional[str]=None,
    scaling_factor: float=1.0
):
    
    if autoencoder_name is not None:
        b, v, c, h, w = latents.shape
        latents = rearrange(latents, "b v c h w -> (b v) c h w ")
        
        if autoencoder_name == "va":
            latents = (latents * latent_std[0].to(latents.device)) + latent_mean[0].to(latents.device) 
        else:
            latents = (1 / scaling_factor) * latents    
        
        if autoencoder_name == "dc":
            with torch.autocast(enabled=False, device_type="cuda", dtype=torch.bfloat16):
                image = autoencoder[view_type].decode(latents).sample 
        elif autoencoder_name == "video_dc":
            with torch.autocast(enabled=False, device_type="cuda", dtype=torch.bfloat16):
                latents = rearrange(latents, "(b v) c h w -> b v c h w", v=v)
                image = autoencoder[view_type].decode(latents)
                image = rearrange(image, "b v c h w -> (b v) c h w")
        elif autoencoder_name == "wan":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                latents = rearrange(latents, "(b v) c h w -> b v c h w", v=v)
                t = len(torch.split(latents, dim=1, split_size_or_sections=5))
                latents = rearrange(latents, "b (t v) c h w -> (b t) v c h w", t=t)
                image = autoencoder[view_type].decode(latents)
                image = rearrange(image, "(b t) v c h w -> b (t v) c h w", t=t)
                image = rearrange(image, "b v c h w -> (b v) c h w")
        elif autoencoder_name == "wan_single":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                latents = rearrange(latents, "(b v) c h w -> b v c h w", v=v)

                image = []
                for i in range(v):
                    latents_single = latents[:, i:i+1, :, :, :]
                    image.append(autoencoder[view_type].decode(latents_single))
                image = torch.concat(image, dim=1)
                image = rearrange(image, "b v c h w -> (b v) c h w")
        else:
            image = autoencoder[view_type].decode(latents).sample 

        image = rearrange(image, "(b v) c h w -> b v c h w", b=b)
    else:
        image = latents
    return (image / 2 + 0.5).clamp(0, 1)    


def get_latents(
    autoencoder: Autoencoder, 
    inputs: Float[Tensor, "b v c h w"], 
    view_type: str, 
    precomputed_latents: dict[str, bool],
    autoencoder_name: Optional[str]=None,
    scaling_factor: float=1.0,
):
        
    if autoencoder_name is None:
        return 2 * inputs["latent"] - 1
    
    if precomputed_latents[view_type]:
        if autoencoder_name == "va":
            latents = (inputs["latent"] - latent_mean.to(inputs["latent"].device)) / latent_std.to(inputs["latent"].device)
        else:
            latents = inputs["latent"] * scaling_factor

    else:
        latents = first_stage_encode(
            autoencoder=autoencoder, 
            inputs=inputs["latent"], 
            view_type=view_type, 
            autoencoder_name=autoencoder_name,
            scaling_factor=scaling_factor
        )
        
    return latents

def get_images(
    autoencoder: Autoencoder, 
    inputs: Float[Tensor, "b v c h w"], 
    view_type: str, 
    precomputed_latents: dict[str, bool],
    autoencoder_name: Optional[str]=None,
    scaling_factor: float=1.0,
):
    if autoencoder_name is None:
        return  inputs["latent"]  
    
    if precomputed_latents[view_type]:
        images = last_stage_decode(
            autoencoder=autoencoder,
            latents=get_latents(
                autoencoder=autoencoder, 
                inputs=inputs, 
                view_type=view_type, 
                precomputed_latents=precomputed_latents,
                autoencoder_name=autoencoder_name,
                scaling_factor=scaling_factor
            ), 
            view_type=view_type,
            autoencoder_name=autoencoder_name,
            scaling_factor=scaling_factor
        )
    else:
        images = inputs["latent"]
    
    return images



def step(
    model: Denoiser, 
    x_t: Float[Tensor, "batch view channel height width"], 
    ts: Float[Tensor, "batch view"], 
    target_pose: CameraInputs, 
    scheduler: Scheduler,
    cond_state: Optional[Float[Tensor, "batch _ _"]]=None, 
    temporal_downsample: int=1,
    cfg_scale: float=1.0
):
    
    b, v_t, *_ = x_t.shape 
    x_t_inputs = scheduler.scale_model_input(x_t, ts)
    
    t = (ts * scheduler.num_train_timesteps - 1).clip(min=0)

    inputs = x_t_inputs.clone()

    # Conditional Forward Pass
    if cond_state is None:
        cond_state = torch.zeros((b, model.num_scene_tokens, model.cond_dim), device=inputs.device)
        cond_state = model.cnd_proj(cond_state)
        cond_state_uc = cond_state
    else:
        cond_state = model.cnd_proj(cond_state)


    denoiser_input = DenoiserInputs(
        view=inputs, 
        pose=target_pose, 
        timestep=t, 
        state=cond_state
    )
    # print(temporal_downsample, inputs.shape, t.shape, cond_state.shape, target_pose.extrinsics.shape)
    pred_conditional, qk_list = model._forward(inputs=denoiser_input, temporal_downsample=temporal_downsample)
    if cfg_scale > 1.0:
        cond_state_uc = model.null_tokens.expand(b, model.num_scene_tokens, -1)            
        denoiser_input.state = cond_state_uc
        pred_unconditional, _ = model._forward(inputs=denoiser_input, temporal_downsample=temporal_downsample)
        pred_out = pred_unconditional + cfg_scale * (pred_conditional - pred_unconditional)

    else:
        pred_out = pred_conditional

    sch_out = scheduler.step(pred_out, ts, x_t).prev_sample


    return sch_out, qk_list, pred_conditional

def latent_to_original_index(latent_idx, temporal_downsample, chunk_index_gap, offset):
    if latent_idx % chunk_index_gap==0:
        idx = temporal_downsample  * latent_idx - (temporal_downsample-1)*(latent_idx//chunk_index_gap)
    else:
        idx = temporal_downsample  * latent_idx - (temporal_downsample-1)*(latent_idx//chunk_index_gap + offset)
    return idx

def original_to_latent_index(idx, temporal_downsample, chunk_index_gap):
    frame_index_gap = temporal_downsample * (chunk_index_gap-1) + 1
    k = idx // frame_index_gap
    r = idx % frame_index_gap
    return chunk_index_gap * k + (r + temporal_downsample-1) // temporal_downsample

def preprocess_denoise_mask(denoise_mask, temporal_downsample, num_latents, chunk_index_gap, offset, autoencoder_name: Optional[str]=None):
    new_denoise_mask = denoise_mask
    if autoencoder_name is not None:
        if autoencoder_name == "wan":
            num_pose = (num_latents // 5) * 17
            idx = torch.nonzero(denoise_mask).squeeze(1)
            start = latent_to_original_index(idx[0], temporal_downsample, chunk_index_gap, offset)
            end = latent_to_original_index(idx[-1]+1, temporal_downsample, chunk_index_gap, offset)
            range_idx = torch.arange(start, end)
            new_denoise_mask = torch.zeros((num_pose,), device=denoise_mask.device, dtype=torch.bool)
            new_denoise_mask[range_idx] = True
        elif autoencoder_name == "video_dc":
            new_denoise_mask = repeat(denoise_mask, "n -> (n t)", t=temporal_downsample)
    return new_denoise_mask

@torch.no_grad()
def sample(
    model: Denoiser,
    x_t: Float[Tensor, "batch view channel height width"], 
    target_pose: CameraInputs, 
    cond_state: torch.Tensor, 
    sampler: Sampler, 
    scheduler: Scheduler,
    autoencoder: Autoencoder,
    temporal_downsample: int=1,
    cfg_scale: float=1.0,
    autoencoder_name: Optional[str]=None,
    scaling_factor: float=1.0,
    chunk_index_gap: int=4,
    offset: int=0, 
    ):

    
    device = x_t.device
    b, v_t, c, h, w = x_t.shape


    pbar = tqdm(range(sampler.global_steps), desc=f"Sampling ({sampler.cfg.name}): ")
    for m in pbar:
        
        pbar.set_postfix({
            "Status:": f" {sampler.current_frame(m)+1}/{sampler.total_frames}"
        })
        
        ts, denoise_mask = sampler(m)
        ts_next, _ = sampler(m+1)

        ts = repeat(ts, "v -> b v", b=b).to(x_t.device)
        ts_next = repeat(ts_next, "v -> b v", b=b).to(x_t.device)
        scheduler.set_scheduling_matrix(ts_next[:, denoise_mask])
        
        # Project latent --> view indices
        # For Wan, num_views = (num_latents // 5) * 17 
        #   - We encode 17 = 1 + 16 = 1 + 4*4 (first frame as it is and the next 16 frames compressed by 4x)
        #   - Yields 5 latents
        #   - Usually 2 chunks (34 views) to allow independent chunked noise levels for rendering

        # For VideoDC, num_views = num_latents * 4 (
        #   - 4x compression
        #   - Local receptive is "mostly" limited to 4 frames
        #   - Can easily restructure view poses
            
        new_denoise_mask = preprocess_denoise_mask(
            denoise_mask=denoise_mask, 
            temporal_downsample=temporal_downsample, 
            num_latents=v_t, 
            chunk_index_gap=chunk_index_gap,
            offset=offset,
            autoencoder_name=autoencoder_name
        )


        # Denoise within the sliding window
        x_t[:, denoise_mask], qk_list, pred_conditional = step(
            model=model, 
            x_t=x_t[:, denoise_mask], 
            ts=ts[:, denoise_mask], 
            target_pose=target_pose[:, new_denoise_mask], 
            cond_state=cond_state, 
            temporal_downsample=temporal_downsample,
            scheduler=scheduler,
            cfg_scale=cfg_scale
        )
        scheduler.unset_scheduling_matrix()

    uncertainty_map = pred_conditional.norm(dim=2) 
    if autoencoder_name is not None:
        if autoencoder_name == "video_dc":
            decoded = last_stage_decode(
                    autoencoder=autoencoder,
                    latents=x_t, 
                    view_type="target",
                    autoencoder_name=autoencoder_name,
                    scaling_factor=scaling_factor,
                )
        elif autoencoder_name == "wan":
            decoded = []
            for x in torch.split(x_t, split_size_or_sections=5, dim=1):
                decoded.append(last_stage_decode(
                    autoencoder=autoencoder,
                    latents=x, 
                    view_type="target",
                    autoencoder_name=autoencoder_name,
                    scaling_factor=scaling_factor,
                ))
            decoded = torch.concat(decoded, dim=1)

    else:
        decoded = (x_t + 1) / 2

    return decoded, uncertainty_map