

import os
import torch
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from jaxtyping import Float, Bool
from einops import repeat, rearrange
from typing import Any, Dict, Iterator, Optional

from cleanfid import fid
from typing import Literal
from torch.nn import Parameter
from torch import Tensor, optim, nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only

from .metrics import Metric
from .denoiser import get_denoiser
from .scheduler import get_scheduler
from .compressor import get_compressor
from .autoencoder import get_autoencoder
from .diffusion import get_images, get_latents, sample
from .scene_generator import get_scene_generator, SceneGenerator
from .types import CameraInputs, CompressorInputs, SceneGeneratorInputs
from .config import ModelCfg, OptimizerCfg, TestCfg, TrainCfg, FreezeCfg, ValCfg
from .sampler import SamplerListCfg, Sampler, get_sampler, SamplerCfg, FullSequenceSampler, FullSequenceSamplerCfg
from ..dataset import DatasetCfg
from ..misc.step_tracker import StepTracker
from ..misc.wandb_tools import log_tensor_as_video
from ..misc.torch_utils import freeze, convert_to_buffer
from ..misc.image_io import prep_image, save_image_video
from ..misc.batch_utils import repeat, preprocess_batch, repeat_batch
from ..misc.mask_utils import generate_random_context_mask, generate_random_context_mask_tail_decay
from ..visualization.layout import  hcat, vcat
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.spline import interpolate_extrinsics_batched


class SceneWrapper(LightningModule):
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
        print("(Main Model) Optimize DDP: ", model_cfg.optimize_ddp)
        print("(Main Model) Using Plucker Coordinates: ", model_cfg.use_plucker)
        print("(Main Model) Using Compressor: ", True if model_cfg.compressor is not None else False)
        print("(Main Model) Using SRT Ray Encoding: ", self.model_cfg.srt_ray_encoding)
        print("(Main Model) Using Standard Ray Encoding: ", model_cfg.use_ray_encoding)
        print("(Main Model) Using EMA: ", self.model_cfg.ema)
        print("(Main Model) Using Memory Efficient Attention: ", self.model_cfg.enable_xformers_memory_efficient_attention)
        
        num_target_split = 1
        if self.dataset_cfg.view_sampler.name in ["video_va_videodc", "evaluation_video"]:
            num_target_split = self.dataset_cfg.view_sampler.num_target_split
        print("(Main Model) Number of Target Splits: ", num_target_split)
        print(f"(Sampler) Timestep (Scene Scheduler) {self.model_cfg.scene_scheduler.kwargs.timestep_shift}")
        print(f"(Sampler) Timestep (Renderer) {self.model_cfg.scheduler.kwargs.timestep_shift}")

        self.sampler = get_sampler(sampler_cfg)
        self.scene_sampler = FullSequenceSampler(FullSequenceSamplerCfg(name="full_sequence"))
        self.override_applied = False
        if model_cfg.autoencoders is not None:
            # print("(Main Model) Using scale factor (Context): ", model_cfg.autoencoders.context.kwargs.scaling_factor)
            # print("(Main Model) Using scale factor (Target): ", model_cfg.autoencoders.target.kwargs.scaling_factor)

            _dict = {}
            if model_cfg.autoencoders.context is not None:
                _dict["context"] = get_autoencoder(model_cfg.autoencoders.context)
            if model_cfg.autoencoders.target is not None:
                _dict["target"] = get_autoencoder(model_cfg.autoencoders.target)

            self.autoencoder = nn.ModuleDict(_dict)

        
        if model_cfg.compressor is not None:
            if model_cfg.autoencoders.context is not None:
                in_channels = model_cfg.autoencoders.context.kwargs.latent_channels
            else:
                in_channels = 3
            self.compressor = get_compressor(model_cfg.compressor, 
                in_channels=in_channels, 
                num_views=self.dataset_cfg.view_sampler.num_context_views, 
                temporal_downsample=1
            )
            cond_dim = self.compressor.output_dim
            self.num_scene_tokens = self.compressor.num_scene_tokens
        else:
            cond_dim = 64   
            self.num_scene_tokens = 256
        
        temporal_downsample = 1
        if getattr(model_cfg.autoencoders, "target") is not None:
            if getattr(self.model_cfg.autoencoders, "target").name == "video_dc":
                temporal_downsample = 4
            
            if not model_cfg.force_incorrect and getattr(self.model_cfg.autoencoders, "target").name == "wan":
                temporal_downsample = 4

        
        self.denoiser = get_denoiser(
            model_cfg.denoiser, 
            cond_dim=cond_dim, 
            num_scene_tokens=self.num_scene_tokens, 
            num_views=self.dataset_cfg.view_sampler.num_target_views, 
            temporal_downsample=temporal_downsample, 
            using_wan=True if "wan" in getattr(self.model_cfg.autoencoders, "target").name else False
        )

        self.scheduler = get_scheduler(model_cfg.scheduler)

        self.scene_generator = get_scene_generator(
            model_cfg.scene_generator, 
            num_scene_tokens=self.num_scene_tokens, 
            cond_dim=cond_dim, 
            temporal_downsample=1 # Input view is an individual image/latent 
        )
        self.scene_scheduler = get_scheduler(model_cfg.scene_scheduler)
        self.metric = Metric().eval()
        self.predicted = []
        self.generated = []

        freeze(self.metric)
        print("(Main Model) Freezing Denoiser")
        freeze(self.denoiser)
        print("(Main Model) Freezing Compressor")
        freeze(self.compressor)
        print("(Main Model) Freezing Autoencoder")
        freeze(self.autoencoder)
        print("(Main Model) Converting to buffer Autoencoder")
        convert_to_buffer(self.autoencoder, persistent=False)
        convert_to_buffer(self.compressor, persistent=False)
        convert_to_buffer(self.denoiser, persistent=False)
        convert_to_buffer(self.metric, persistent=False)


        if self.model_cfg.ema:
            self.ema = AveragedModel(
                self.scene_generator, 
                multi_avg_fn=get_ema_multi_avg_fn(0.995)
            )
                
        self.validation_type = None
        self.test_step_outputs = []
        self.tokens = []

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
        # Always allow missing/unexpected keys
        super().load_state_dict(state_dict, strict=False)

    def on_train_batch_start(self, batch, batch_idx):
        step = self.global_step
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
            
    def set_timesteps(self, num: Optional[int]=None, name: str="validation"):
        """
            Args:
                num (Optional[int]): Override the number of inference steps. Default to None.
        """
        num_inference_timesteps = self.model_cfg.scene_scheduler.num_inference_steps if num is None else num
        print(f"Setting Max Timesteps for {name} to: ", num_inference_timesteps)
        self.scene_scheduler.set_timesteps(num_inference_timesteps)    

         
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
    
    def get_noise_level(self, shape: tuple=(1,), dtype: torch.dtype = torch.float):
        
        if self.model_cfg.scene_scheduler.name == "rectified_flow":
            if self.model_cfg.scene_scheduler.kwargs.weighting == "uniform":
                timesteps = torch.rand(
                    size=shape, 
                    device=self.device,
                    dtype=dtype
                ) 
            elif self.model_cfg.scene_scheduler.kwargs.weighting == "logit_normal":
                timesteps = torch.normal(0.0, 1.0,
                    size=shape, 
                    device=self.device,
                    dtype=dtype
                ).sigmoid_()
            elif self.model_cfg.scene_scheduler.kwargs.weighting == "shifted":
                timesteps = torch.rand(
                    size=shape, 
                    device=self.device,
                    dtype=dtype
                ) 
                timesteps = self.scene_scheduler.shift_timestep(timestep=timesteps, shift=self.model_cfg.scene_scheduler.kwargs.timestep_shift)

            else:
                raise NotImplementedError(f"{self.model_cfg.scene_scheduler.kwargs.weighting} weighting is not implemented!")
            
            
        else:
            timesteps = torch.randint(
                0, 
                self.model_cfg.scene_scheduler.num_train_timesteps, 
                size=shape, 
                device=self.device,
                dtype=torch.long
            ) 
        return timesteps
    

    def generate_random_cond_mask(self, b: int, v: int) -> torch.Tensor:
        # Step 1: Generate random boolean mask
        mask = torch.rand(b, v) < 0.5

        # Force the first entry in each row to be True
        mask[:, 0] = True

        return mask

    def rescale_timesteps(self, timestep):
        
        if self.model_cfg.scene_scheduler.name == "rectified_flow":

            # if self.model_cfg.denoiser.name == "lightningdit":
            #     t = 1 - timestep
            # else:
            t = (timestep * self.model_cfg.scene_scheduler.num_train_timesteps - 1).clip(min=0)
        elif self.model_cfg.denoiser.name == "dfot":
            t_min = torch.atan(torch.exp(-0.5 * torch.tensor(15.0)))
            t_max = torch.atan(torch.exp(-0.5 * torch.tensor(-15.0)))
            t = 0.125 * (-2 * torch.log(torch.tan(torch.tensor(t_min + timestep * (t_max - t_min)))) + 2 * torch.log(torch.tensor(0.125)))

        else:
            t = timestep
        return t

    
    def preprocess_scene_tokens(self, scene_tokens, shape, device):
        
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
        return scene_tokens
    
    def process_gt(self, latents, noise, timestep):
        
        if self.model_cfg.scene_scheduler.kwargs.prediction_type == "epsilon":
            target = noise
        
        elif self.model_cfg.scene_scheduler.kwargs.prediction_type == "v_prediction":
            target = self.scene_scheduler.get_velocity(latents, noise, timestep)
        
        elif self.model_cfg.scene_scheduler.kwargs.prediction_type == "sample":
            target = latents
        
        elif self.model_cfg.scene_scheduler.kwargs.prediction_type == "flow":

            target = self.scene_scheduler.get_flow(latents, noise)
        else:
            raise NotImplementedError()
        
        return target


    def step_scene(
        self, 
        model: SceneGenerator, 
        x_t: Float[Tensor, "batch num dim"], 
        ts: Float[Tensor, "batch"], 
        anchor_pose: CameraInputs,
        cond_latents: Float[Tensor, "batch view channels height width"],
        cond_pose: CameraInputs,
        cond_mask: Bool[Tensor, "batch cond_view"]
    ):
        
        b, v_t, *_ = x_t.shape 
        x_t_inputs = self.scene_scheduler.scale_model_input(x_t, ts)
        
        t = self.rescale_timesteps(ts)
        
        inputs = x_t_inputs.clone()
       


        generator_input = SceneGeneratorInputs(
            view=cond_latents, 
            pose=cond_pose, 
            anchor_pose=anchor_pose,
            timestep=t, 
            state=inputs
        )
        pred_conditional, qk_list = model._forward(inputs=generator_input, cond_mask=cond_mask)
        if self.model_cfg.use_cfg:
            uncond_mask = torch.zeros_like(cond_mask, dtype=torch.bool, device=cond_mask.device)
            pred_unconditional, _ = model._forward(inputs=generator_input, cond_mask=uncond_mask)
            pred_out = pred_unconditional + self.test_cfg.scene_cfg * (pred_conditional - pred_unconditional)

        else:
            pred_out = pred_conditional

        sch_out = self.scene_scheduler.step(pred_out, ts, x_t).prev_sample


        return sch_out, qk_list, pred_conditional

    @torch.no_grad()
    def sample_scene(
        self, 
        cond_latents: Float[Tensor, "batch view channels height width"],
        anchor_pose: CameraInputs,
        cond_pose: CameraInputs,
        num_cond: int=1
    ):

        b = anchor_pose.extrinsics.shape[0]
        device = anchor_pose.extrinsics.device

        x_t = torch.randn((b, self.num_scene_tokens, self.compressor.output_dim), device=device)

        if self.model_cfg.use_ema_sampling:
            print("Loading EMA weights")
            # self.ema.copy_to(self.denoiser)
            model = self.ema
        else:
            model = self.scene_generator
        cond_mask = torch.zeros((b, self.dataset_cfg.view_sampler.max_cond_number), device=device, dtype=torch.bool)
        cond_mask[:, :num_cond] = True
        pbar = tqdm(range(self.scene_sampler.global_steps), desc=f"Sampling Scene with {num_cond} conditioning: ")
        for m in pbar:

            
            ts, denoise_mask = self.scene_sampler(m)
            ts_next, _ = self.scene_sampler(m+1)
            
            ts = repeat(ts, "n -> b n", b=b).to(device)
            ts_next = repeat(ts_next, "n -> b n", b=b).to(device)

            self.scene_scheduler.set_scheduling_matrix(ts_next)
            # Denoise within the sliding window
            x_t, _, _ = self.step_scene(
                model=model, 
                x_t=x_t, 
                ts=ts, 
                anchor_pose=anchor_pose, 
                cond_latents=cond_latents,
                cond_pose=cond_pose,
                cond_mask=cond_mask
            )
            # atten_argmax = self.get_argmax_scene(qk_list, height=h, width=w)
            # print(atten_argmax.shape)
            self.scene_scheduler.unset_scheduling_matrix()

        return x_t


    def generate_batch_with_scene(self, batch, sampler: Sampler, repeat_factor: int=1, num_cond: int=1, scene_tokens=None, predict_scene: bool=True):
        
        if scene_tokens is None:
            cond_latents = get_latents(
                autoencoder=self.autoencoder,
                inputs=batch["cond"],
                view_type="context",
                precomputed_latents=self.dataset_cfg.precomputed_latents,
                autoencoder_name=getattr(self.model_cfg.autoencoders, "context").name,
                scaling_factor=getattr(self.model_cfg.autoencoders, "context").kwargs.scaling_factor,
            )
        else:
            cond_latents = batch["cond"]["latent"]
        batch["target"] = repeat_batch(batch["target"], repeat_factor)
        device = cond_latents.device
        dtype = cond_latents.dtype
        # b, v_c, *_ = context_latents.shape
        b, v_t, *_ = batch["target"]["extrinsics"].shape
        
        if getattr(self.model_cfg.autoencoders, "target").name == "video_dc":
            temporal_downsample = 4
        else:
            temporal_downsample = 1
        num = (v_t // temporal_downsample) * temporal_downsample
        batch["target"]["extrinsics"] = batch["target"]["extrinsics"][:, :num]
        batch["target"]["intrinsics"] = batch["target"]["intrinsics"][:, :num]


        if self.model_cfg.autoencoders.target.name in ["kl"]:
            c = self.model_cfg.autoencoders.target.kwargs.latent_channels // 2
        else:
            c = self.model_cfg.autoencoders.target.kwargs.latent_channels


        h, w = self.model_cfg.denoiser.input_shape
        x_t = torch.randn((b, num // temporal_downsample, c, h, w)).to(device)  
        print(x_t.shape, batch["target"]["extrinsics"].shape)
        x_t *= self.scheduler.init_noise_sigma 


        anchor_pose=CameraInputs(
            intrinsics=batch["context"]["intrinsics"],
            extrinsics=batch["context"]["extrinsics"]
        )
        cond_pose=CameraInputs(
            intrinsics=batch["cond"]["intrinsics"],
            extrinsics=batch["cond"]["extrinsics"]
        )
        target_pose = CameraInputs(
            intrinsics=batch["target"]["intrinsics"],
            extrinsics=batch["target"]["extrinsics"]
        )

        self.scene_sampler.set_scheduling_matrix(
            horizon=self.num_scene_tokens,
            steps=self.model_cfg.scene_scheduler.num_inference_steps, 
            concurrency=self.num_scene_tokens, 
            device=device,
            dtype=dtype,
            cond_mask_indices=None
        )
        if self.model_cfg.scene_scheduler.kwargs.timestep_shift is not None:
            self.scene_sampler.shift_scheduling_matrix(shift=self.model_cfg.scene_scheduler.kwargs.timestep_shift)
            print(f"(Sampler) Shifting scheduling matrix for scene generator by {self.model_cfg.scene_scheduler.kwargs.timestep_shift}")

        scene_tokens_gen = self.sample_scene(cond_latents=cond_latents, anchor_pose=anchor_pose, cond_pose=cond_pose, num_cond=num_cond)
        scene_tokens_gen /= self.model_cfg.scene_generator.scale_factor

        if scene_tokens is None and predict_scene:
            context_latents=get_latents(
                autoencoder=self.autoencoder,
                inputs=batch["context"],
                view_type="context",
                precomputed_latents=self.dataset_cfg.precomputed_latents,
                autoencoder_name=getattr(self.model_cfg.autoencoders, "context").name,
                scaling_factor=getattr(self.model_cfg.autoencoders, "context").kwargs.scaling_factor
            )
            context_inputs = CompressorInputs(
                view=context_latents,
                pose=anchor_pose
            )

            tokens_pred, _ = self.compressor._forward(inputs=context_inputs)
            if self.model_cfg.compressor.scene_token_projection == "kl":
                scene_tokens_pred = tokens_pred.sample()
            else:
                scene_tokens_pred = tokens_pred
        else:
            scene_tokens_pred = scene_tokens    



        scene_tokens_gen = repeat(scene_tokens_gen, "b ... -> (b n) ...", n=repeat_factor)
        if predict_scene:
            scene_tokens_pred = repeat(scene_tokens_pred, "b ... -> (b n) ...", n=repeat_factor)


        
        sampler.set_scheduling_matrix(
            horizon=num // temporal_downsample,
            steps=self.model_cfg.scheduler.num_inference_steps, 
            concurrency=self.dataset_cfg.view_sampler.num_target_views, 
            device=device,
            dtype=dtype,
            cond_mask_indices=None
        )

        if self.model_cfg.scheduler.kwargs.timestep_shift is not None:
            sampler.shift_scheduling_matrix(shift=self.model_cfg.scheduler.kwargs.timestep_shift)
            print(f"(Sampler) Shifting scheduling matrix for renderer by {self.model_cfg.scheduler.kwargs.timestep_shift}")

        if predict_scene:
            predicted_outputs = (*sample(
                model=self.denoiser,
                x_t=x_t.clone(), 
                target_pose=target_pose,
                cond_state=scene_tokens_pred,
                sampler=sampler,
                scheduler=self.scheduler,
                autoencoder=self.autoencoder,
                temporal_downsample=temporal_downsample,
                cfg_scale=self.model_cfg.cfg_scale,
                autoencoder_name=getattr(self.model_cfg.autoencoders, "target").name,
                scaling_factor=getattr(self.model_cfg.autoencoders, "target").kwargs.scaling_factor,
                chunk_index_gap=self.dataset_cfg.view_sampler.chunk_index_gap,
                offset=self.dataset_cfg.view_sampler.offset
            ), scene_tokens_pred)
        else:
            predicted_outputs = (None, None, None)
        return ((*sample(
            model=self.denoiser,
            x_t=x_t.clone(), 
            target_pose=target_pose,
            cond_state=scene_tokens_gen,
            sampler=sampler,
            scheduler=self.scheduler,
            autoencoder=self.autoencoder,
            temporal_downsample=temporal_downsample,
            cfg_scale=self.model_cfg.cfg_scale,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "target").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "target").kwargs.scaling_factor,
            chunk_index_gap=self.dataset_cfg.view_sampler.chunk_index_gap,
            offset=self.dataset_cfg.view_sampler.offset
        ), scene_tokens_gen), predicted_outputs)

    def training_step(self, batch, batch_idx):
        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
            self.log(f"step_tracker/step", self.step_tracker.get_step())

        b, v_cond, *_ = batch["cond"]["extrinsics"].shape

        batch["context"]["extrinsics"] = torch.concat([batch["cond"]["extrinsics"], batch["context"]["extrinsics"]], dim=1)

        batch = preprocess_batch(batch, index=0)
        batch["cond"]["extrinsics"] = batch["context"]["extrinsics"][:, :v_cond]
        batch["context"]["extrinsics"] = batch["context"]["extrinsics"][:, v_cond:]

        cond_latents = batch["cond"]["latent"]
        device = cond_latents.device
        dtype = cond_latents.dtype
        anchor_pose = CameraInputs(
            intrinsics=batch["context"]["intrinsics"],
            extrinsics=batch["context"]["extrinsics"]
        )

        cond_latents = get_latents(
            autoencoder=self.autoencoder,
            inputs=batch["cond"], 
            view_type="context",
            precomputed_latents=self.dataset_cfg.precomputed_latents,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "context").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "context").kwargs.scaling_factor,
        )
        context_latents = get_latents(
            autoencoder=self.autoencoder,
            inputs=batch["context"], 
            view_type="context",
            precomputed_latents=self.dataset_cfg.precomputed_latents,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "context").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "context").kwargs.scaling_factor,
        )
        context_inputs = CompressorInputs(
            view=context_latents,
            pose=anchor_pose
        )
        with torch.no_grad():
            tokens, *_ = self.compressor(inputs=context_inputs)
        
        if self.model_cfg.compressor.scene_token_projection == "kl":
            scene_tokens = tokens.sample().detach() * self.model_cfg.scene_generator.scale_factor
        else:
            scene_tokens = tokens.detach() * self.model_cfg.scene_generator.scale_factor

        
        # Add noise to target views
        noise = torch.randn_like(scene_tokens, device=device)  

        timestep = self.get_noise_level((b, ), dtype=dtype)[:, None].expand(-1, self.num_scene_tokens)

        noisy_tokens = self.scene_scheduler.add_noise(scene_tokens, noise, timestep)
         
        t = self.rescale_timesteps(timestep=timestep)   
        b, v_cond, *_ = cond_latents.shape
        cond_mask = self.generate_random_cond_mask(b, v_cond).to(device)

        conditional = torch.ones((b,), dtype=torch.bool, device=device)
        if self.train_cfg.cfg_train:
            # Randomly choose to train conditionally or unconditionally
            conditional = np.random.choice([True, False], b, p=[0.90, 0.10])
            conditional = torch.tensor(conditional, device=device, dtype=torch.bool)
        cond_mask *= conditional[:, None]
        cond_mask = cond_mask.to(torch.bool)
        # Denoise
        generator_input = SceneGeneratorInputs(
            view=cond_latents, 
            pose=CameraInputs(
                intrinsics=batch["cond"]["intrinsics"],
                extrinsics=batch["cond"]["extrinsics"]
            ), 
            anchor_pose=anchor_pose,
            timestep=t, 
            state=noisy_tokens
        )

        pred, _ = self.scene_generator(inputs=generator_input, cond_mask=cond_mask)
        
        gt = self.process_gt(scene_tokens, noise, timestep)

        loss = F.mse_loss(pred, gt, reduction="none").mean()

       
        if self.global_rank == 0:
            print(
                f"Train step {self.step_tracker.get_step()}; "
                # f"scene = {batch['scene']}; "
                # f"context = {batch['context']['index'].tolist()}; "
                # f"target = {batch['target']['index'].tolist()}; "
                f"loss = {loss.item():.4f}"
            )
        # Diffusion Loss
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
            f"Cond = {batch['cond']['index'].tolist()}; "
            f"Context = {batch['context']['index'].tolist()}; "
            # f"Target = {batch['target']['index'].tolist()}; "
            f"Rank = {self.global_rank}; "
        )
        
        cond_views = get_images(
            autoencoder=self.autoencoder,
            inputs=batch["cond"], 
            view_type="context",
            precomputed_latents=self.dataset_cfg.precomputed_latents,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "context").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "context").kwargs.scaling_factor
        )

        device = cond_views.device
        dtype = cond_views.dtype
        b, v_cond, *_ = cond_views.shape
        b, v_c, *_ = batch["context"]["extrinsics"].shape
        b, v_cond, *_ = batch["cond"]["extrinsics"].shape
        

        batch["context"]["extrinsics"] = torch.concat([batch["cond"]["extrinsics"], batch["context"]["extrinsics"]], dim=1)
        batch = preprocess_batch(batch, index=0)
        batch["cond"]["extrinsics"] = batch["context"]["extrinsics"][:, :v_cond]
        batch["context"]["extrinsics"] = batch["context"]["extrinsics"][:, v_cond:]


        batch_vis = []
        batch_scene = []
        for j in range(b):
            scene = batch["scene"][j]

            cond_vis = add_label(hcat(*[cond_views[j, i, ...] for i in range(v_cond)]), "Cond Views")
            # context_vis = add_label(hcat(*[context_views[j, i, ...] for i in range(v_c)]), "Context Views")
            
            vis = add_label(vcat(cond_vis), scene)

            batch_vis.append(prep_image(vis))
            batch_scene.append(scene)
            
        self.logger.log_image(
            f"Conditioning and Context Views",
            batch_vis,
            step=val_step,
            caption=batch_scene,
        )
       
        if self.dataset_cfg.name == "all_scene":
            scene_tokens = batch["scene_tokens"]
        else:
            scene_tokens = None

        t = self.val_cfg.video_length

        b, v_c, *_= batch["context"]["extrinsics"].shape
        *_, c, h, w = batch["cond"]["latent"].shape
        target_extrinsics = interpolate_extrinsics_batched(batch["context"]["extrinsics"], self.val_cfg.video_length - v_c + 1)
        indices = torch.linspace(0, t, steps=t, device=device)
        new_target = {
            "extrinsics": target_extrinsics.to(device),
            "intrinsics": batch["context"]["intrinsics"][:, 0:1].expand(-1, t, -1, -1).clone(),
            "latent": torch.zeros((b, t, c, h, w), device=device).to(dtype),
            "index": indices[None].expand(b, -1)
        }
        batch["target"] = new_target      
    
        # try:    
        (
            (sampled_views_gen, uncertainty_map_gen, scene_tokens_gen), 
            (sampled_views_pred, uncertainty_map_pred, scene_tokens_pred)
        ) = self.generate_batch_with_scene(
            batch, 
            self.sampler, 
            num_cond=1, # Only single-view for validation 
            scene_tokens=scene_tokens
        )

        # uncertainty_min = uncertainty_map.reshape(b, v_t, -1).min(dim=-1)
        # uncertainty_max = uncertainty_map.reshape(b, v_t, -1).max(dim=-1)

        # uncertainty_map = (uncertainty_map - uncertainty_min[..., None, None]) / (uncertainty_max[..., None, None] - uncertainty_min[..., None, None])


        self.sampler.log_vis(self.logger, "video", self.step_tracker)
        self.scene_sampler.log_vis(self.logger, "scene_sampler", self.step_tracker)
        log_tensor_as_video(self.logger, sampled_views_gen, f"Context Interpolation (Generated) ({self.sampler.cfg.name})", fps=24, step=val_step, caption=batch["scene"])        
        log_tensor_as_video(self.logger, sampled_views_pred, f"Context Interpolation (Predicted) ({self.sampler.cfg.name})", fps=24, step=val_step, caption=batch["scene"])        
        # log_tensor_as_video(self.logger, uncertainty_vis, f"Video ({sampler.cfg.name}) Uncertainty", fps=8, step=self.step_tracker.get_step(), caption=f"{batch_scene}")        
        self.generated.append((sampled_views_gen * 255).clamp(0, 255).to(torch.uint8).cpu())
        self.predicted.append((sampled_views_pred * 255).clamp(0, 255).to(torch.uint8).cpu())
        torch.cuda.empty_cache()
        # except:
        #     pass
        return None
    
    def on_validation_epoch_end(self):

        step = self.step_tracker.get_step()
        val_step = (step+1) // self.val_check_interval
        # if len(self.predicted) == 0 or len(self.generated) == 0:
        #     return
        sampled_views = torch.concat(self.predicted)
        target_views = torch.concat(self.generated)
        # --- sequence-level metrics ---
        gathered_sampled = self.all_gather(sampled_views)
        gathered_target = self.all_gather(target_views)

        # flatten across world size
        gathered_sampled = rearrange(gathered_sampled, "... (k v) c h w -> (... k) v c h w", v=16)
        gathered_target = rearrange(gathered_target, "... (k v) c h w -> (... k) v c h w", v=16)

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
        self.predicted.clear()
        self.generated.clear()

        print("Setting Max Timesteps for training to: ", self.model_cfg.scene_scheduler.num_train_timesteps)
        self.scene_scheduler.set_timesteps(self.model_cfg.scene_scheduler.num_train_timesteps)
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

        b, v_c, *_ = batch["context"]["extrinsics"].shape
        b, v_t, *_ = batch["target"]["extrinsics"].shape

        
        batch = preprocess_batch(batch, index=0)

        batch["cond"] = {
            "extrinsics": batch["context"]["extrinsics"].clone()[:, [0, -1, -1]],
            "intrinsics": batch["context"]["intrinsics"].clone()[:, [0, -1, -1]],
            "latent": batch["context"]["latent"].clone()[:, [0, -1, -1]],  
            "index": batch["context"]["index"].clone()[:, [0, -1, -1]]
        }
        device = batch["target"]["extrinsics"].device
        ctxt_idx = torch.linspace(0, v_t - 1, 12, device=device).long()
        batch["context"]["extrinsics"] = batch["target"]["extrinsics"].clone()[:, ctxt_idx]
        batch["context"]["intrinsics"] = batch["target"]["intrinsics"].clone()[:, ctxt_idx]

        cond_views = get_images(
            autoencoder=self.autoencoder,
            inputs=batch["cond"],
            view_type="context",
            precomputed_latents=self.dataset_cfg.precomputed_latents,
            autoencoder_name=getattr(self.model_cfg.autoencoders, "context").name,
            scaling_factor=getattr(self.model_cfg.autoencoders, "context").kwargs.scaling_factor,
        )

            
        (
            (sampled_views_gen, uncertainty_map_gen, scene_tokens_gen), 
            (sampled_views_pred, uncertainty_map_pred, scene_tokens_pred)
        ) = self.generate_batch_with_scene(
            batch, 
            self.sampler, 
            num_cond=self.test_cfg.num_condition_views, 
            repeat_factor=1,
            predict_scene=False
        )
                    
        for j in tqdm(range(b), desc="Saving (Gen) Sampled Views: "):
            save_image_video(
                images=sampled_views_gen[j], 
                indices=torch.arange(0, sampled_views_gen[j].shape[0]), 
                output_dir=self.output_dir / "predicted"/ batch["scene"][j],
                name=self.sampler.cfg.name + "_gen", save_img=True, save_video=False
            )

                
        for j in tqdm(range(b), desc="Saving Original Views: "):
            save_image_video(
                images=target_views[j], 
                indices=batch["target"]["index"][j], 
                output_dir=self.output_dir/ "gt" / batch["scene"][j] ,
                name="original", save_img=True, save_video=False
            )

            save_image_video(
                images=context_views[j], 
                indices=batch["context"]["index"][j], 
                output_dir=self.output_dir / "context" /  batch["scene"][j],
                name="context", save_video=False
            )

            save_image_video(
                images=cond_views[j, :self.test_cfg.num_condition_views], 
                indices=batch["cond"]["index"][j], 
                output_dir=self.output_dir / "cond"/ batch["scene"][j],
                name="cond", save_video=False
            )
        

        return None

    
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
    
        
    def on_after_backward(self):
        if self.global_step == 0 and self.global_rank == 0:
            print("\n[DEBUG] Checking for parameters with grad=None after backward:")
            for name, p in self.named_parameters():
                if p.requires_grad and p.grad is None:
                    print("  UNUSED PARAM (no grad):", name)
            print("[DEBUG] End unused-param scan\n")
        # scan all grads for NaN or Inf
        for p in self.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    print("Skipping Nans!")
                    self.log("nan_grad_skipped", 1.0, prog_bar=True)
                    # zero out everything—this makes the upcoming optimizer.step() a no-op
                    for q in self.parameters():
                        q.grad = None
                    break  # done
    
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
        param_list = [{"params": self.scene_generator.parameters()}]

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
