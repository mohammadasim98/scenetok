import torch

from torch import Tensor
from dataclasses import dataclass
from jaxtyping import Float, Bool
from typing import Union, Literal, Optional

from .sampler import Sampler

@dataclass
class PyramidSamplerCfg:
    name: Literal["pyramid"]
    
    clean_targets: int=1
    uncertainty_scale: int=1
    
class PyramidSampler(Sampler[PyramidSamplerCfg]):


    def get_scheduling_matrix(
        self,
        horizon: int,
        steps: int, 
        concurrency: int, 
        device: torch.device,
        dtype: torch.dtype,
        cond_mask_indices: Optional[list[int]]=None,
        clean_targets: Optional[int]=None
    ) -> Union[Float[Tensor, "batch num view"], Bool[Tensor, "batch num view"]]:
        
        clean_targets = self.cfg.clean_targets if clean_targets is None else clean_targets
        window_shift = concurrency - clean_targets
        mat_list = []
        mask_list = [[True if i < concurrency else False for i in range(horizon) ]]
        
        m = mask_list[-1]
        
        window_position = 0
        previous_window_position = window_position
        if cond_mask_indices is None:
            x = [steps for _ in range(horizon)]
        else:
            x = [steps if i not in cond_mask_indices else 0 for i in range(horizon)]
        mat_list.append(x.copy())
        while sum(x) != 0:
            prev_idx = 0
            for idx in range(horizon):
                if x[idx] >= 0 and (idx - window_position) < concurrency:
                    if idx == 0:
                        x[idx] -= 1
                    elif (x[prev_idx] <= steps - self.cfg.uncertainty_scale) or (previous_window_position != window_position):
                        x[idx] -= 1
                        previous_window_position = window_position

                prev_idx = idx
            x = [x[i] if x[i] >= 0 else 0 for i in range(horizon)] 
            windowed_x = x[window_position:window_position + concurrency]
            null_mask = [True if i < len(windowed_x) and windowed_x[i] == 0 else False for i in range(concurrency)]     
            if self.cfg.uncertainty_scale > 0 and sum(null_mask) == self.cfg.clean_targets+1:
                window_position += 1
                if window_position < horizon - concurrency:
                    m = [True if (window_position <= i < (window_position + concurrency)) else False for i in range(horizon)]
                else:
                    m = [True if i >= horizon - concurrency else False for i in range(horizon)]
            elif (self.cfg.uncertainty_scale == 0 or self.cfg.uncertainty_scale == steps) and sum(x[window_position:window_position + concurrency]) == 0:
                if sum(x[window_position:window_position + concurrency]) == 0:
                    window_position += window_shift
                    if window_position < horizon - concurrency:
                        m = [True if (window_position <= i < (window_position + concurrency)) else False for i in range(horizon)]
                    else:
                        m = [True if i >= horizon - concurrency else False for i in range(horizon)]

            mat_list.append(x.copy())
            mask_list.append(m)
        return torch.tensor(mat_list, device=device, dtype=dtype) / steps, torch.tensor(mask_list, device=device, dtype=torch.bool)
