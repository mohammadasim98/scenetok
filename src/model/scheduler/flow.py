

import math
import torch
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional, Union, List
from src.misc.tensor import unsqueeze_as
from .noise_schedules import cosine_simple_diffusion_schedule
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin


# Refer to https://github.com/huggingface/diffusers/blob/89e4d6219805975bd7d253a267e1951badc9f1c0/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
@dataclass
class RectifiedFlowMatchingKwargsCfg:
    weighting: Literal["logit_normal", "uniform", "shifted"] = "uniform"
    num_train_timesteps: int = 1000
    prediction_type: Literal["epsilon", "flow"] = "flow"
    timestep_shift: float | None = None
    schedule_type: Literal["cosine", "linear"] = "linear"
@dataclass
class RectifiedFlowMatchingSchedulerCfg:
    name: Literal["rectified_flow"]
    num_train_timesteps: int
    num_inference_steps: int
    pretrained_from: str | None
    sampling_type: Literal["random_uniform", "random_independent", "random_chunked_uniform"]
    kwargs: RectifiedFlowMatchingKwargsCfg

@dataclass
class RectifiedFlowMatchingSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None



class RectifiedFlowMatchingScheduler(SchedulerMixin, ConfigMixin):
    """
    `DDPMScheduler` explores the connections between denoising score matching and Langevin dynamics sampling.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        t_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        t_end (`float`, defaults to 0.02):
            The final `beta` value.

    """


    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        prediction_type: Literal["epsilon", "flow"] = "flow",
        weighting: Literal["logit_normal", "uniform"] = "uniform",
        timestep_shift: float | None = None,
        schedule_type: Literal["cosine", "linear"] = "linear",
        num_chunks: int=1
    ):
        print("(Scheduler) Using rectified flow")
        print("(Scheduler) Using timestep weighting: ", weighting)
        print("(Scheduler) Using prediction type: ", prediction_type)
        print("(Scheduler) Using timestep shift: ", timestep_shift)
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.timestep_shift = timestep_shift
        self.schedule_type = schedule_type
        self.scheduling_matrix = None
        self.num_chunks = num_chunks
        if self.schedule_type == "linear":
            self.timesteps = torch.linspace(0.0, 1.0, num_train_timesteps + 1).flip(-1)
        if schedule_type == "cosine":
            raise NotImplementedError()
        
        if self.timestep_shift is not None:
            self.timesteps = self.timestep_shift * self.timesteps / (1 + (self.timestep_shift - 1) * self.timesteps)

        # timesteps = self.sigmas * num_train_timesteps
        self.init_noise_sigma = 1.0

        # self.sigma_min = self.sigmas[-1].item()
        # self.sigma_max = self.sigmas[0].item()

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        return sample
    
    def set_scheduling_matrix(self, matrix):
    
        self.scheduling_matrix = matrix
    
    def unset_scheduling_matrix(self):

        self.scheduling_matrix = None
    
    def shift_timestep(self, timestep, shift):
        
        return shift * timestep / (1 + (shift - 1) * timestep)
    
    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps
    
    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.


        """
        self.timesteps = torch.linspace(0.0, 1.0, num_inference_steps + 1).flip(-1)
        # if self.timestep_shift is not None:
        #     print("Timestep: ", self.timesteps)
        #     self.timesteps = self.timestep_shift * self.timesteps / (1 + (self.timestep_shift - 1) * self.timesteps)
        #     print("Shifted Timestep: ", self.timesteps)
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: float,
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[RectifiedFlowMatchingSchedulerOutput]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """

        timestep_next = self.next_timestep(timestep)
 
        timestep = unsqueeze_as(timestep, model_output).to(model_output.device)
        timestep_next = unsqueeze_as(timestep_next, model_output).to(model_output.device)


        # sigma_next = sigma_next.to(model_output.device)
        # sigma = sigma.to(model_output.device)
        
        step_size = timestep_next - timestep

        if self.prediction_type == "flow":
            flow = model_output
        
        elif self.prediction_type == "epsilon":

            flow = (model_output - sample) / (1 - timestep)
        else:
            raise NotImplementedError()


        pred_prev_sample = sample + step_size * flow
        pred_original_sample = sample - timestep * flow

        return RectifiedFlowMatchingSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:

        timesteps = unsqueeze_as(timesteps, original_samples)
        noisy_samples = (1 - timesteps) * original_samples + timesteps * noise
        
        return noisy_samples

    def get_flow(self, sample: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as sample

        return noise - sample

    def __len__(self):
        return self.config.num_train_timesteps

    def next_timestep(self, timestep):
        
        
        if self.scheduling_matrix is None:
            indices = (self.timesteps == timestep).nonzero()[0].item()
            if indices < len(self.timesteps):
                return self.timesteps[indices + 1]
            else:
                return self.timesteps[indices]
            
        else:
            return self.scheduling_matrix


