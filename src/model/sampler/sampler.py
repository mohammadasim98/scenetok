import io
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch import nn, Tensor
from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar, Union
from jaxtyping import Float, Bool
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from lightning.pytorch.loggers.wandb import WandbLogger

from ...misc.image_io import prep_image
from ...misc.step_tracker import StepTracker

T = TypeVar("T")


class Sampler(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(
        self, 
        cfg: T
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.scheduling_matrix = None
        self.denoising_matrix = None
        self.concurrency = None
        
    def get_scheduling_matrix(
        self,
    ) -> Union[Float[Tensor, "batch num view"], Bool[Tensor, "batch num view"]]:
        
        raise NotImplementedError()    
    

    
    def set_scheduling_matrix(
        self,
        **kwargs
    ):
        self.scheduling_matrix, self.denoising_matrix = self.get_scheduling_matrix(**kwargs)
    
    def shift_scheduling_matrix(self, shift):
        self.scheduling_matrix = shift * self.scheduling_matrix / (1 + (shift - 1) * self.scheduling_matrix)
    def center_skew(self, eps: float=0.004):
        # x = torch.linspace(0.0, 1.0, 100)

        self.scheduling_matrix = self.scheduling_matrix.logit(eps=eps)
        self.scheduling_matrix = (self.scheduling_matrix - self.scheduling_matrix.min()) / (self.scheduling_matrix.max() - self.scheduling_matrix.min())


    def forward(
        self,
        index: int
    ) -> Union[Float[Tensor, "batch view"], Bool[Tensor, "batch view"]]:
        
        return self.scheduling_matrix[index], self.denoising_matrix[index]
    
    @property
    def global_steps(self):
        return self.scheduling_matrix.shape[0] - 1
    
    def current_frame(self, index):
        return (self.scheduling_matrix[index] == 0.0).sum()
    
    @property
    def total_frames(self):
        return self.scheduling_matrix.shape[1]
    
    def get_visualization(self):
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_title("Denoising Window")
        cmap = ListedColormap(['white', 'green'])
        im = ax[0].imshow(self.denoising_matrix.cpu().float(), cmap=cmap, aspect='auto')
        legend_elements = [
            Patch(facecolor='green', label=f"Window size: {self.concurrency}"),
        ]
        ax[0].set_ylabel('Global Step')
        ax[0].legend(handles=legend_elements, loc='upper right')
        fig.colorbar(im, ax=ax[0], ticks=[0.0, 1.0])
        ax[1].set_title("Scheduling Matrix")
        im = ax[1].imshow(self.scheduling_matrix.cpu().float(), cmap='viridis', aspect='auto')
        ax[1].set_xlabel('Frame')

        fig.colorbar(im, ax=ax[1], ticks=[0.0, 1.0], label='Noise Level')
        ax[1].set_xlabel('Frame')
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        # Load as PIL Image
        image = Image.open(buf).convert("RGB")

        # Optionally: close the figure to free memory
        plt.close(fig)
        # Now `image` is a PIL.Image object

        return np.asarray(image)

            
    def log_vis(self, logger: WandbLogger, name: str, step: StepTracker):
        
        vis = self.get_visualization()
        logger.log_image(
            f"{name}/sampling_{self.cfg.name})",
            [vis],
            step=step.get_step(),
            caption=[f""],
        )