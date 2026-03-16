
import torch
import einops
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from typing import Literal
from torch.nn import Parameter
from torch import optim, nn
from einops import repeat, rearrange
from lightning.pytorch import LightningModule
from typing import Any, Dict, Iterator, Optional
from lightning.pytorch.loggers.wandb import WandbLogger
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from .metrics import Metric
from .denoiser import get_denoiser
from .scheduler import get_scheduler
from .compressor import get_compressor
from .autoencoder import get_autoencoder
from .diffusion import get_images, get_latents, sample
from .types import CameraInputs, CompressorInputs, DenoiserInputs
from .sampler import SamplerListCfg, Sampler, get_sampler, SamplerCfg
from .config import ModelCfg, OptimizerCfg, TestCfg, TrainCfg, FreezeCfg, ValCfg
from ..dataset import DatasetCfg
from ..misc.step_tracker import StepTracker
from ..misc.wandb_tools import log_tensor_as_video
from ..misc.image_io import prep_image, save_image_video
from ..misc.torch_utils import freeze, convert_to_buffer
from ..misc.batch_utils import repeat, sequence_concatenate, preprocess_batch, repeat_batch, sequence_reverse
from ..misc.mask_utils import generate_random_context_mask, generate_random_context_mask_tail_decay, generate_biased_boolean_mask, random_mask_biased
from ..visualization.layout import  hcat
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.spline import interpolate_extrinsics_batched


class DiffusionWrapper(LightningModule):
    logger: Optional[WandbLogger]
    model_cfg: ModelCfg
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    freeze_cfg: FreezeCfg
    step_tracker: StepTracker | None
    output_dir: Path | None = None

    def __init__(
        self,
        model_cfg: ModelCfg,
        dataset_cfg: DatasetCfg,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        val_cfg: ValCfg,
        sampler_cfg: SamplerListCfg | SamplerCfg,
        freeze_cfg: FreezeCfg,
        batch_size: int,
        step_tracker: StepTracker | None,
        output_dir: Path | None = None,
        val_check_interval: int=5000,
        mode: Literal["train", "val", "test", "predict_train", "predict_test"]="train",
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.dataset_cfg = dataset_cfg
        self.model_cfg = model_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.val_cfg = val_cfg
        self.freeze_cfg = freeze_cfg
        self.step_tracker = step_tracker
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.val_check_interval = val_check_interval
        self.mode = mode
        torch._dynamo.config.optimize_ddp = model_cfg.optimize_ddp
        # if bool(int(os.getenv("DEBUG", 0))):
        #     print("Anomaly detection is enabled during DEBUG")
        #     torch.autograd.set_detect_anomaly(False)

        # self.automatic_optimization = False

        print("(Main Model) Using Plucker Coordinates: ", model_cfg.use_plucker)
        print("(Main Model) Using Compressor: ", True if model_cfg.compressor is not None else False)
        print("(Main Model) Using SRT Ray Encoding: ", self.model_cfg.srt_ray_encoding)
        print("(Main Model) Using Standard Ray Encoding: ", model_cfg.use_ray_encoding)
        print("(Main Model) Using EMA: ", self.model_cfg.ema)
        print("(Main Model) Using Memory Efficient Attention: ", self.model_cfg.enable_xformers_memory_efficient_attention)
        print("(Main Model) Using Scheduler from: ", self.model_cfg.scheduler.pretrained_from)
       
        num_target_split = self.model_cfg.denoiser.num_target_split
        
        print("(Main Model) Number of Target Splits: ", num_target_split)
        print(f"(Sampler) Timestep {self.model_cfg.scheduler.kwargs.timestep_shift}")
        print(f"(Sampler) Clean Targets {sampler_cfg.clean_targets}")

        self.sampler = get_sampler(sampler_cfg)
        self.override_applied = False
        if model_cfg.autoencoders is not None:
            _dict = {}
            if model_cfg.autoencoders.context is not None:
                _dict["context"] = get_autoencoder(model_cfg.autoencoders.context)
            if model_cfg.autoencoders.target is not None:
                _dict["target"] = get_autoencoder(model_cfg.autoencoders.target)

            self.autoencoder = nn.ModuleDict(_dict)

        
        if model_cfg.compressor is not None:
            if getattr(model_cfg.autoencoders, "context") is not None:
                in_channels = model_cfg.autoencoders.context.kwargs.latent_channels
            else:
                in_channels = 3
            self.compressor = get_compressor(
                model_cfg.compressor, 
                in_channels=in_channels, 
                num_views=self.dataset_cfg.view_sampler.num_context_views, 
                temporal_downsample=1
            )
            cond_dim = self.compressor.output_dim
            num_scene_tokens = self.compressor.num_scene_tokens
        else:
            cond_dim = 64   
            num_scene_tokens = 256
        
        temporal_downsample = 1
        if getattr(model_cfg.autoencoders, "target") is not None:
            if getattr(self.model_cfg.autoencoders, "target").name == "video_dc":
                temporal_downsample = 4
            
            if not model_cfg.force_incorrect and getattr(self.model_cfg.autoencoders, "target").name == "wan":
                temporal_downsample = 4
        
        self.denoiser = get_denoiser(
            model_cfg.denoiser, 
            cond_dim=cond_dim, 
            num_scene_tokens=num_scene_tokens, 
            num_views=self.dataset_cfg.view_sampler.num_target_views, 
            temporal_downsample=temporal_downsample, 
            using_wan=True if "wan" in getattr(self.model_cfg.autoencoders, "target").name else False)
        self.scheduler = get_scheduler(model_cfg.scheduler)

        if self.freeze_cfg.denoiser:
            print("(Main Model) Freezing Denoiser")
            freeze(self.denoiser)
        if self.freeze_cfg.compressor:
            print("(Main Model) Freezing Compressor")
            freeze(self.compressor)
        if self.freeze_cfg.autoencoder and self.model_cfg.autoencoders is not None:
            print("(Main Model) Freezing Autoencoder")
            freeze(self.autoencoder)

        if self.model_cfg.autoencoders is not None:
            print("(Main Model) Converting to buffer Autoencoder")
            convert_to_buffer(self.autoencoder, persistent=False)
        
        self.metric = Metric().eval()
        freeze(self.metric)
        convert_to_buffer(self.metric, persistent=False)


        if self.model_cfg.ema:
            self.ema = AveragedModel(
                self.denoiser, 
                multi_avg_fn=get_ema_multi_avg_fn(0.995)
            )
                
        self.validation_type = None
        self.frozen_compressor = self.freeze_cfg.compressor
        self.unfrozen_compressor = False
        self.frozen_scene_query = False
        self.test_step_outputs = []
        self.predicted = []
        self.generated = []



    def on_before_zero_grad(self, *args, **kwargs):
        if self.model_cfg.ema:
            self.ema.update_parameters(self.denoiser)   

    def setup(self, stage: str) -> None:
        # Scale base learning rates to effective batch size
        if stage == "fit":
            # assumes one fixed batch_size for all train dataloaders!
            effective_batch_size = self.trainer.accumulate_grad_batches \
                * self.trainer.num_devices \
                * self.trainer.num_nodes \
                * self.batch_size

            self.lr = effective_batch_size * self.optimizer_cfg.lr \
                if self.optimizer_cfg.scale_lr else self.optimizer_cfg.lr
        return super().setup(stage)



    def load_state_dict(self, state_dict, strict: bool = True):
        super().load_state_dict(state_dict, strict=False)

    
    def set_timesteps(self, num: Optional[int]=None, name: str="validation"):
        """
            Args:
                num (Optional[int]): Override the number of inference steps. Default to None.
        """
        num_inference_timesteps = self.model_cfg.scheduler.num_inference_steps if num is None else num
        print(f"Setting Max Timesteps for {name} to: ", num_inference_timesteps)
        self.scheduler.set_timesteps(num_inference_timesteps)    

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.set_timesteps(name="validation")
    
    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        self.set_timesteps(name="testing")
    
    def get_conditioning_mask(self, shape, device, dtype):
        if self.model_cfg.compressor is not None:
            
            context_mask = generate_random_context_mask(shape=shape, device=device).to(dtype)
        else:
            context_mask = generate_random_context_mask_tail_decay(shape=shape, device=device).to(dtype)
        
        return context_mask
    
    def get_noise_level(self, shape: tuple=(1,), dtype: torch.dtype = torch.float, mu: float=0.0, sigma: float=1.0):
        
        if self.model_cfg.scheduler.name == "rectified_flow":
            if self.model_cfg.scheduler.kwargs.weighting == "uniform":
                timesteps = torch.rand(
                    size=shape, 
                    device=self.device,
                    dtype=dtype
                ) 
            elif self.model_cfg.scheduler.kwargs.weighting == "logit_normal":
                timesteps = torch.normal(mu, sigma,
                    size=shape, 
                    device=self.device,
                    dtype=dtype
                ).sigmoid_()
            elif self.model_cfg.scheduler.kwargs.weighting == "shifted":
                timesteps = torch.rand(
                    size=shape, 
                    device=self.device,
                    dtype=dtype
                ) 
                timesteps = self.scheduler.shift_timestep(timestep=timesteps, shift=self.model_cfg.scheduler.kwargs.timestep_shift)

            else:
                raise NotImplementedError(f"{self.model_cfg.scheduler.kwargs.weighting} weighting is not implemented!")
            
            
        else:
            timesteps = torch.randint(
                0, 
                self.model_cfg.scheduler.num_train_timesteps, 
                size=shape, 
                device=self.device,
                dtype=torch.long
            ) 
        return timesteps
    
    def rescale_timesteps(self, timestep):
        
        if self.model_cfg.scheduler.name == "rectified_flow":
            t = (timestep * self.model_cfg.scheduler.num_train_timesteps - 1).clip(min=0)
        else:
            t = timestep
        return t

    
    def preprocess_scene_tokens(self, scene_tokens, shape, device, token_mask=None):
        
        if scene_tokens is None:
            if self.model_cfg.compressor is None:
                scene_tokens = torch.zeros(shape, device=device)
                scene_tokens = self.denoiser.cnd_proj(scene_tokens)
            else:
                if self.model_cfg.no_null_expand:
                    scene_tokens = self.denoiser.null_tokens.expand(shape[0], -1, -1)
                else:
                    scene_tokens = self.denoiser.null_tokens.expand(*shape[:2], -1)
        else:
            scene_tokens = self.denoiser.cnd_proj(scene_tokens)
        if token_mask is not None:
            scene_tokens[~token_mask] = self.denoiser.null_tokens[0].to(scene_tokens.dtype)
        return scene_tokens
    
    def process_gt(self, latents, noise, timestep):
        
        if self.model_cfg.scheduler.kwargs.prediction_type == "epsilon":
            target = noise
        
        elif self.model_cfg.scheduler.kwargs.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timestep)
        
        elif self.model_cfg.scheduler.kwargs.prediction_type == "sample":
            target = latents
        
        elif self.model_cfg.scheduler.kwargs.prediction_type == "flow":

            target = self.scheduler.get_flow(latents, noise)
        else:
            raise NotImplementedError()
        
        return target



    @torch.no_grad()
    def generate_batch_with_scene(self, batch, sampler: Sampler, repeat_factor: int=1):
        
        context_latents = get_latents(
            autoencoder=self.autoencoder,
            inputs=batch["context"], 
            view_type="context",
            precomputed_latents=self.dataset_cfg.precomputed_latents,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "context").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "context").kwargs.scaling_factor,
        )

        target = repeat_batch(batch["target"], repeat_factor)
        device = context_latents.device
        dtype = context_latents.dtype
        b_c, v_c, *_ = context_latents.shape
        b, v_t, *_ = target["extrinsics"].shape

        if getattr(self.model_cfg.autoencoders, "target") is not None:
            if self.model_cfg.autoencoders.target.name in ["kl"]:
                c = self.model_cfg.autoencoders.target.kwargs.latent_channels // 2
            else:
                c = self.model_cfg.autoencoders.target.kwargs.latent_channels
        else:
            c = 3

        temporal_downsample = 1
        if getattr(self.model_cfg.autoencoders, "target") is not None:
            if getattr(self.model_cfg.autoencoders, "target").name == "video_dc":
                temporal_downsample = 4
                num = (v_t // temporal_downsample)
                target["extrinsics"] = target["extrinsics"][:, :num*temporal_downsample]
                target["intrinsics"] = target["intrinsics"][:, :num*temporal_downsample]
            
            elif getattr(self.model_cfg.autoencoders, "target").name == "wan":
                temporal_downsample = 4
                num = (v_t // 17) * 5

                target["extrinsics"] = target["extrinsics"][:, :(num//5)*17]
                target["intrinsics"] = target["intrinsics"][:, :(num//5)*17]

        h, w = self.model_cfg.denoiser.input_shape
        x_t = torch.randn((b, num, c, h, w)).to(device)  
        x_t *= self.scheduler.init_noise_sigma 
        
        context_camera = CameraInputs(
            intrinsics=batch["context"]["intrinsics"],
            extrinsics=batch["context"]["extrinsics"]
        )


        target_pose = CameraInputs(
            intrinsics=target["intrinsics"],
            extrinsics=target["extrinsics"]
        )

        context_mask = None

        if not self.model_cfg.mask_context:
            context_inputs = CompressorInputs(
                view=context_latents,
                pose=context_camera,
                mask=None
            )
        else:
            context_inputs = CompressorInputs(
                view=context_latents[:, ~context_mask[0]],
                pose=context_camera[:, ~context_mask[0]],
                mask=None
            )

        tokens, qks = self.compressor._forward(inputs=context_inputs)
        

        if self.model_cfg.compressor.scene_token_projection == "kl":
            scene_tokens = tokens.sample()
        else:
            scene_tokens = tokens

        scene_tokens = repeat(scene_tokens, "b ... -> (b n) ...", n=repeat_factor)
        
        sampler.set_scheduling_matrix(
            horizon=num,
            steps=self.model_cfg.scheduler.num_inference_steps, 
            concurrency=self.dataset_cfg.view_sampler.num_target_views, 
            device=device,
            dtype=dtype,
            cond_mask_indices=None
        )
        if self.model_cfg.scheduler.kwargs.weighting == "shifted":
            print(f"(Sampler) Shifting scheduling matrix by {self.model_cfg.scheduler.kwargs.timestep_shift}")
            sampler.shift_scheduling_matrix(self.model_cfg.scheduler.kwargs.timestep_shift)
        sampler.log_vis(self.logger, step=self.step_tracker, name=f"({sampler.cfg.name})")
        print("Shape of latents: ", x_t.shape)
        return *sample(
            model=self.denoiser,
            x_t=x_t, 
            target_pose=target_pose,
            cond_state=scene_tokens,
            sampler=sampler,
            scheduler=self.scheduler,
            autoencoder=self.autoencoder,
            temporal_downsample=temporal_downsample,
            cfg_scale=self.model_cfg.cfg_scale,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "target").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "target").kwargs.scaling_factor,
            chunk_index_gap=self.dataset_cfg.view_sampler.chunk_index_gap,
            offset=self.dataset_cfg.view_sampler.offset, 
        ), scene_tokens
    
    def training_step(self, batch, batch_idx):
        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
            self.log(f"step_tracker/step", self.step_tracker.get_step())

        # convert all camera poses for context and target to relative w.r.t a random context camera
        # during test time, you can select any context index to be the origin
        # during training of scenegen, make sure the origin is present and that the first conditioning
        # image starts at the origin 
        batch = preprocess_batch(batch)

        # Get latents if input is rgb otherwise scale latents by the scaling_factor
        # In case of VA-VAE, scale and shift by the predefined latent statistics
        target_latents = get_latents(
            autoencoder=self.autoencoder,
            inputs=batch["target"], 
            view_type="target",
            precomputed_latents=self.dataset_cfg.precomputed_latents,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "target").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "target").kwargs.scaling_factor,
        )
        
        device = target_latents.device
        dtype = target_latents.dtype
        b, v_t, c, h, w = target_latents.shape

        # CFG if enabled
        conditional_tokens = True
        if self.train_cfg.cfg_train and not self.freeze_cfg.denoiser:
            # Randomly choose to train conditionally or unconditionally
            conditional_tokens = np.random.choice([True, False], 1, p=[0.90, 0.10])
        
        token_mask = None
        if conditional_tokens and self.model_cfg.compressor is not None:
            # Get latents if input is rgb otherwise scale latents by the scaling_factor
            # In case of VA-VAE, scale and shift by the predefined latent statistics
            context_latents = get_latents(
                autoencoder=self.autoencoder,
                inputs=batch["context"], 
                view_type="context",
                precomputed_latents=self.dataset_cfg.precomputed_latents,
                autoencoder_name=getattr(self.model_cfg.autoencoders, "context").name,
                scaling_factor=getattr(self.model_cfg.autoencoders, "context").kwargs.scaling_factor,
            )
            
            b, v_c, *_ = context_latents.shape
            # Experimental: masking context views until min context views
            if self.model_cfg.mask_context and np.random.choice([True, False], p=[0.4, 0.6]):
                context_mask = generate_biased_boolean_mask((b, v_c), self.dataset_cfg.view_sampler.min_context_views).to(context_latents.device)
            else:
                context_mask = None

            context_inputs = CompressorInputs(
                view=context_latents,
                pose=CameraInputs(
                    intrinsics=batch["context"]["intrinsics"],
                    extrinsics=batch["context"]["extrinsics"]
                ),
                mask=context_mask
            )
            if self.frozen_compressor:
                with torch.no_grad():
                    tokens, *_ = self.compressor(inputs=context_inputs)
            else:
                tokens, *_ = self.compressor(inputs=context_inputs)
            
            if self.model_cfg.compressor.scene_token_projection == "kl":
                scene_tokens = tokens.sample()
            else:
                scene_tokens = tokens


            # Experimental: Add noise to tokens
            if self.model_cfg.noisy_scene_tokens and np.random.choice([True, False], p=[self.model_cfg.noise_prob, 1-self.model_cfg.noise_prob]):
                scene_noise = torch.randn_like(scene_tokens, device=device)  
                timestep_scene = self.get_noise_level((b, self.compressor.num_scene_tokens), dtype=dtype, mu=self.model_cfg.mu, sigma=self.model_cfg.sigma)
                scene_tokens = self.scheduler.add_noise(scene_tokens, scene_noise, timestep_scene)

            # Experimental: Masking tokens
            if self.model_cfg.mask_tokens:
                token_mask, ratios, num_false = random_mask_biased(B=scene_tokens.shape[0], N=scene_tokens.shape[1], M=0.6, device="cpu")


        else:
            # for unconditional sampling (rendering)
            scene_tokens = None
        
        
        # Define noise-level per tokens
        if self.model_cfg.scheduler.sampling_type == "random_uniform":
            timestep_shape = (b, )
        elif self.model_cfg.scheduler.sampling_type == "random_chunked_uniform":
            timestep_shape = (b, self.dataset_cfg.view_sampler.num_target_split)
        elif self.model_cfg.scheduler.sampling_type == "random_independent":
            timestep_shape = (b, v_t)
        else:
            raise NotImplementedError(f"Sampling type in scheduler is not correctly specified and instead got {self.model_cfg.scheduler.sampling_type}")
        
        # Allow targets with same noise-levels
        if np.random.choice([True, False], p=[0.2, 0.8]) and self.model_cfg.enforce_uniform_noise:
            timestep_shape = (b, )
        
        # Get noise-level
        timestep = self.get_noise_level(timestep_shape, dtype=dtype)
        # Repeat timestep in case of (chunked) uniform noise-levels
        if timestep.ndim == 1:
            timestep = repeat(timestep, "b -> b v", v=v_t)

        elif self.model_cfg.scheduler.sampling_type == "random_chunked_uniform":
            timestep = repeat(timestep, "b n -> b (n v)", v=self.dataset_cfg.view_sampler.num_target_views // self.dataset_cfg.view_sampler.num_target_split)
        
        # Experimental: Force zero noise-levels for conditioning 
        if self.model_cfg.force_clean:
            target_cond_mask = self.get_conditioning_mask((b, v_t), device=device, dtype=dtype) 
            timestep = timestep * target_cond_mask
        
        # Sample Noise
        noise = torch.randn_like(target_latents, device=device)  

        # Add noise to targets
        noisy_latents = self.scheduler.add_noise(target_latents, noise, timestep)

        # If masking tokens, then define mask either before or after up-projection layer
        # Otherwise no masking, and simply up-project the tokens for rendering
        # Parameters of this up-projection is part of the denoiser
        scene_tokens = self.preprocess_scene_tokens(
            scene_tokens=scene_tokens, 
            shape=(b, self.denoiser.num_scene_tokens, self.denoiser.cond_dim), 
            device=device, 
            token_mask=token_mask
        )
        
        t = self.rescale_timesteps(timestep=timestep)   
            
        # Denoise
        denoiser_input = DenoiserInputs(
            view=noisy_latents, 
            pose=CameraInputs(
                intrinsics=batch["target"]["intrinsics"],
                extrinsics=batch["target"]["extrinsics"]
            ), 
            timestep=t, 
            state=scene_tokens

        )
        
        # flow prediction for targets
        pred, _ = self.denoiser(
            inputs=denoiser_input, 
            temporal_downsample=self.dataset_cfg.view_sampler.temporal_downsample
        )
        
        # Get ground truth flow for targets
        gt = self.process_gt(target_latents, noise, timestep)

        # Loss
        loss = F.mse_loss(pred, gt, reduction='none')

        # Only apply on noisy targets
        if self.model_cfg.force_clean:
            loss = einops.reduce(loss, "b v c h w -> b v", "mean")
            loss = loss * target_cond_mask
            loss = loss.sum(-1) / target_cond_mask.sum(-1)
        else:
            loss = einops.reduce(loss, "b v c h w -> b", "mean")

        # Apply KL divergence weighting and scheduling
        kl_weight = 0.0
        if self.model_cfg.compressor.scene_token_projection == "kl" and conditional_tokens and self.model_cfg.compressor is not None and not self.frozen_compressor:
            kl = tokens.kl()
            if self.global_step <= self.model_cfg.compressor.kl_schedule[0]:
                kl_weight = self.model_cfg.compressor.kl_weights[0]
            elif self.global_step <= self.model_cfg.compressor.kl_schedule[1] and self.global_step > self.model_cfg.compressor.kl_schedule[0]:
                t = (self.global_step - self.model_cfg.compressor.kl_schedule[0]) / (self.model_cfg.compressor.kl_schedule[1] - self.model_cfg.compressor.kl_schedule[0])
                kl_weight = (1 - t) * self.model_cfg.compressor.kl_weights[0] + t*self.model_cfg.compressor.kl_weights[1]
            else:
                kl_weight = self.model_cfg.compressor.kl_weights[1]
            loss = loss + kl_weight * kl
            self.log("loss/kl", kl.mean())   
        loss = loss.mean()

        opt = self.optimizers()
        current_lr = opt.param_groups[0]["lr"]
        if self.global_rank == 0:
            print(
                # f"Train step {self.step_tracker.get_step()}; "
                # f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"target = {batch['target']['index'][:, [0, 16, -1]].tolist()}; "
                f"loss = {loss.item():.4f} lr = {current_lr}"
            )
        self.log("loss/diffusion", loss) 

        return loss
    
    # @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx: Optional[int]=None):

        step = self.step_tracker.get_step()
        val_step = (step+1) // self.val_check_interval

        print(
            f"Name = Standard; "
            f"Validation Step {val_step}; "
            f"Step {batch_idx}; "
            f"Scene = {batch['scene']}; "
            f"Context = {batch['context']['index'].tolist()}; "
            f"Target = {batch['target']['index'].tolist()}; "
            f"Rank = {self.global_rank}; "
        )

        # In case if latent inputs e.g., during training with precomputed latents
        context_views = get_images(
            autoencoder=self.autoencoder, 
            inputs=batch["context"], 
            view_type="context", 
            precomputed_latents=self.dataset_cfg.precomputed_latents,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "context").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "context").kwargs.scaling_factor
        )
        target_views = get_images(
            autoencoder=self.autoencoder, 
            inputs=batch["target"], 
            view_type="target", 
            precomputed_latents=self.dataset_cfg.precomputed_latents,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "target").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "target").kwargs.scaling_factor
        )
        b, v_c, *_ = context_views.shape

        # Relative pose w.r.t the middle context index
        batch = preprocess_batch(batch, index=v_c//2)

        # Sample target views
        sampled_views, _, _ = self.generate_batch_with_scene(batch, self.sampler)
        self.generated.append(sampled_views)
        self.predicted.append(target_views)
        # Only do remaining on Rank: 0 in case of multi-gpu/node
        if self.global_rank != 0:
            return None


        b, v_t, c, h, w = sampled_views.shape

        batch_vis = []
        batch_scene = []
        for j in range(b):
            scene = batch["scene"][j]
            batch_scene.append(scene)
            
        log_tensor_as_video(self.logger, sampled_views, f"Sampled Video", fps=8, step=val_step, caption=batch_scene)        
        log_tensor_as_video(self.logger, target_views, f"Original Video", fps=8, step=val_step, caption=batch_scene) 
          
        for j in range(b):
            scene = batch["scene"][j]
            context_vis = add_label(hcat(*[context_views[j, i, ...] for i in range(v_c)]), "Context Views")
            vis = add_label(context_vis, scene)
            batch_vis.append(prep_image(vis))
            
        self.logger.log_image(
            f"Context ({self.sampler.cfg.name})",
            batch_vis,
            step=val_step,
            caption=batch_scene,
        )

        # Do a spline interpolation using context poses as "knot" and sample a video
        print("Generating Context Interpolation Video...")
        if self.val_cfg.video:
            b, v_c, c, h, w = batch["context"]["latent"].shape
            t = self.val_cfg.video_length
            start = batch["context"]["extrinsics"][:, 0]
            target_extrinsics = interpolate_extrinsics_batched(batch["context"]["extrinsics"], self.val_cfg.video_length - v_c + 1)
            indices = torch.linspace(0, t, steps=t, device=start.device)
            new_target = {
                "extrinsics": target_extrinsics.to(start.device),
                "intrinsics": batch["context"]["intrinsics"][:, 0:1].expand(-1, t, -1, -1).clone(),
                "latent": torch.zeros((b, t, c, h, w), device=start.device).to(start.dtype),
                "index": indices[None].expand(b, -1)
            }
            batch["target"] = new_target  
            try:    
                sampled_views, _, _ = self.generate_batch_with_scene(batch, self.sampler)
                log_tensor_as_video(self.logger, sampled_views, f"Context Interpolation ({self.sampler.cfg.name})", fps=24, step=val_step, caption=batch_scene)
            except:
                pass
        torch.cuda.empty_cache()

        return None

    def on_validation_epoch_end(self):
        
        step = self.step_tracker.get_step()
        val_step = (step+1) // self.val_check_interval

        if len(self.predicted) == 0 or len(self.generated) == 0:
            return None
        sampled_views = torch.concat(self.predicted)
        target_views = torch.concat(self.generated)
        print(sampled_views.shape)
        print(target_views.shape)
        metrics = self.metric(sampled_views.flatten(0, 1), target_views.flatten(0, 1), psnr=True, ssim=True, lpips=True)
        self.log("lpips", metrics["lpips"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("ssim", metrics["ssim"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("psnr", metrics["psnr"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        for key, value in metrics.items():
            self.logger.log_metrics({f"full_sequence/{key}": value}, val_step)
        
        # --- sequence-level metrics ---
        gathered_sampled = self.all_gather(sampled_views)
        gathered_target = self.all_gather(target_views)
        num_views = gathered_sampled.shape[-4]
        chunk_size = min(16, num_views)
        if getattr(self.model_cfg.autoencoders, "target") is not None:
            if getattr(self.model_cfg.autoencoders, "target").name == "wan":
                chunk_size = min(17, num_views)

        # flatten across world size
        gathered_sampled = rearrange(gathered_sampled, "... (k v) c h w -> (... k) v c h w", v=chunk_size)
        gathered_target = rearrange(gathered_target, "... (k v) c h w -> (... k) v c h w", v=chunk_size)

        # flatten views for metric
        gathered_sampled = rearrange(gathered_sampled, "... c h w -> (...) c h w")
        gathered_target = rearrange(gathered_target, "... c h w -> (...) c h w")

        if self.global_rank == 0:
            
            gathered_sampled = gathered_sampled.cpu()
            gathered_target = gathered_target.cpu()

            general_metrics = self.metric(gathered_sampled, gathered_target, fvd=True, fid=True)
            
            self.metric.reset_fid()
            self.metric.reset_fvd()
            
            if general_metrics["fid"] is not None:
                self.log("fid", general_metrics["fid"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
            
            if general_metrics["fvd"] is not None:
                self.log("fvd", general_metrics["fvd"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
            
            for key, value in general_metrics.items():
                self.logger.log_metrics({f"{key}": value}, val_step)

        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return None

    def on_validation_end(self):
        step = self.step_tracker.get_step()
        # if step == 0:
        #     return
        if len(self.predicted) != 0:
            self.predicted.clear()
        if len(self.generated) != 0:
            self.generated.clear()

        print("Setting Max Timesteps for training to: ", self.model_cfg.scheduler.num_train_timesteps)
        self.scheduler.set_timesteps(self.model_cfg.scheduler.num_train_timesteps)
        return None

    def test_step(self, batch, batch_idx):
        
        step = self.step_tracker.get_step()
        
        print(
            f"Current epoch {step}; "
            f"Step {batch_idx}; "
            f"Scene = {batch['scene']}; "
            f"Context = {batch['context']['index'].tolist()}; "
            f"Target = {batch['target']['index'].tolist()}; "
        )

        b, v_t, *_ = batch["target"]["extrinsics"].shape
        b, v_c, *_ = batch["context"]["extrinsics"].shape

        print(f"Number of context views: {v_c}")
        print(f"Number of target views: {v_t}")

        target_views=get_images(
            autoencoder=self.autoencoder,
            inputs=batch["target"],
            view_type="target",
            precomputed_latents=self.dataset_cfg.precomputed_latents,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "target").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "target").kwargs.scaling_factor
        )
        context_views=get_images(
            autoencoder=self.autoencoder,
            inputs=batch["context"],
            view_type="context",
            precomputed_latents=self.dataset_cfg.precomputed_latents,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "context").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "context").kwargs.scaling_factor
        )

        # Relative camera w.r.t middle context camera (can be any other context camera)
        batch = preprocess_batch(batch, index=v_c//2)
        sampled_views, _, _ = self.generate_batch_with_scene(
            batch, 
            self.sampler, 
            repeat_factor=1
        )

        for j in tqdm(range(b), desc="Saving Sampled Views: "):
            save_image_video(
                images=sampled_views[j], 
                indices=torch.arange(0, sampled_views[j].shape[0]), 
                output_dir=self.output_dir / "predicted" / batch["scene"][j],
                name=self.sampler.cfg.name, save_img=True, save_video=False
            )
        for j in tqdm(range(b), desc="Saving Original Views: "):
            save_image_video(
                images=target_views[j], 
                indices=torch.arange(0, target_views[j].shape[0]), 
                output_dir=self.output_dir / "gt" / batch["scene"][j],
                name="original", save_img=True, save_video=False
            )

            save_image_video(
                images=context_views[j], 
                indices=batch["context"]["index"][j], 
                output_dir=self.output_dir / "context" / batch["scene"][j],
                name="context", save_img=True, save_video=False
            )
        return None

    def on_train_batch_start(self, batch, batch_idx):
        step = self.global_step
        if step >= self.model_cfg.compressor.freeze_after and self.model_cfg.compressor.freeze_after != -1 and not self.frozen_compressor:
            print(f"[INFO] Freezing Compressor after {step} steps!")
            freeze(self.compressor)
            self.frozen_compressor = True


        if type(self.optimizer_cfg.scheduler) != list:

            warmup_iters = self.optimizer_cfg.scheduler.kwargs.get("total_iters", 0)
            if step < warmup_iters and not self.override_applied:
                self.print(f"[INFO] Warmup not done yet! Current step {step} < {warmup_iters}. Overriding will happen afterwards!")
                self.override_applied = True


            if step >= warmup_iters and not self.override_applied:
                for group in self.trainer.optimizers[0].param_groups:
                    ckpt_lr = group["lr"]
                    if ckpt_lr != self.lr:
                        group["lr"] = self.lr
                        self.print(f"[INFO] Warmup done at step {step}. Overriding LR from {ckpt_lr} to {self.lr}")
                        self.override_applied = True
        else:
            if self.optimizer_cfg.override_lr is not None and not self.override_applied:


                for group in self.trainer.optimizers[0].param_groups:
                    ckpt_lr = group["lr"]
                    if ckpt_lr != self.optimizer_cfg.override_lr:
                        group["lr"] = self.optimizer_cfg.override_lr
                        self.print(f"[INFO] Overriding LR from {ckpt_lr} to {self.optimizer_cfg.override_lr}")
                        self.override_applied = True

    # NOTE: Check for nans in gradients otherwise skip the optimizer step by zeroing out grad
    def on_after_backward(self):
        if self.global_step == 0 and self.global_rank == 0:
            print("\n[DEBUG] Checking for parameters with grad=None after backward:")
            for name, p in self.named_parameters():
                if p.requires_grad and p.grad is None:
                    print("  UNUSED PARAM (no grad):", name)
            print("[DEBUG] End unused-param scan\n")
        
        # DEBUG: Compute and log gradient norms
        grad_norms = []
        total_norm_sq = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                grad_norms.append(param_norm)
                total_norm_sq += param_norm ** 2
        
        if grad_norms:
            total_norm = total_norm_sq ** 0.5
            grad_norms_tensor = torch.tensor(grad_norms)
            avg_norm = grad_norms_tensor.mean().item()
            std_norm = grad_norms_tensor.std().item() if len(grad_norms) > 1 else 0.0
            
            self.log("grad/total_norm", total_norm, on_step=True, on_epoch=False)
            self.log("grad/avg_norm", avg_norm, on_step=True, on_epoch=False)
            self.log("grad/std_norm", std_norm, on_step=True, on_epoch=False)
        
        # scan all grads for NaN or Inf
        for name, p in self.named_parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    print(f"Skipping Nans! in {name}")
                    self.log("nan_grad_skipped", 1.0, prog_bar=True)
                    # zero out everything—this makes the upcoming optimizer.step() a no-op
                    for q in self.parameters():
                        if q.grad is not None: 
                            q.grad.detach().zero_()
                        # q.grad = None
                    break  # done
    @staticmethod
    def get_optimizer(
        optimizer_cfg: OptimizerCfg,
        params: Iterator[Parameter] | list[Dict[str, Any]],
        lr: float
    ) -> optim.Optimizer:
        return getattr(optim, optimizer_cfg.name)(
            params,
            lr=lr,
            **(optimizer_cfg.kwargs if optimizer_cfg.kwargs is not None else {})       
        )
    
    
    @staticmethod
    def get_lr_scheduler(
        opt: optim.Optimizer, 
        optim_cfg: OptimizerCfg
    ) -> optim.lr_scheduler.LRScheduler:
        lr_scheduler_cfg = optim_cfg.scheduler

        if type(lr_scheduler_cfg) == list:
            return optim.lr_scheduler.SequentialLR(
                optimizer=opt,
                schedulers=[
                    getattr(optim.lr_scheduler, cfg.name)(
                        opt,
                        **(cfg.kwargs if cfg.kwargs is not None else {})     
                    ) for cfg in lr_scheduler_cfg
                ],
                milestones=optim_cfg.milestones
            )
        else:
            return getattr(optim.lr_scheduler, lr_scheduler_cfg.name)(
                opt,
                **(lr_scheduler_cfg.kwargs if lr_scheduler_cfg.kwargs is not None else {})     
            )

    def configure_optimizers(self):
        param_list = [{"params": self.denoiser.parameters()}]
        if self.model_cfg.compressor is not None:
            param_list.append({"params": self.compressor.parameters()})
        optimizer = self.get_optimizer(self.optimizer_cfg, param_list, self.lr)
        if self.optimizer_cfg.scheduler is not None:
            if type(self.optimizer_cfg.scheduler) == list:
                frequency = self.optimizer_cfg.scheduler[0].frequency
                interval = self.optimizer_cfg.scheduler[0].interval
            else:
                frequency = self.optimizer_cfg.scheduler.frequency
                interval = self.optimizer_cfg.scheduler.interval
            lr_scheduler_config = {
                "scheduler": self.get_lr_scheduler(optimizer, self.optimizer_cfg),
                "frequency": frequency,
                "interval": interval
            }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
