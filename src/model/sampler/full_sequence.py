import torch

from torch import Tensor
from dataclasses import dataclass
from jaxtyping import Float, Bool
from typing import Union, Literal, Optional

from .sampler import Sampler

@dataclass
class FullSequenceSamplerCfg:
    name: Literal["full_sequence"]
    
    clean_targets: int=0

class FullSequenceSampler(Sampler[FullSequenceSamplerCfg]):


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
        if cond_mask_indices is None:
            x = [steps for _ in range(horizon)]
        else:
            x = [steps if i not in cond_mask_indices else 0 for i in range(horizon)]
        mat_list.append(x.copy())
        while sum(x) != 0:
            
            for idx in range(horizon):
                if x[idx] != 0 and (idx - window_position) < concurrency:
                    x[idx] -= 1 
                    
            if sum(x[window_position:window_position + concurrency]) == 0:
                window_position += window_shift
                if window_position < horizon - concurrency:
                    m = [True if (window_position <= i < (window_position + concurrency)) else False for i in range(horizon)]
                else:
                    m = [True if i >= horizon - concurrency else False for i in range(horizon)]

            mat_list.append(x.copy())
            mask_list.append(m)
        return torch.tensor(mat_list, device=device, dtype=dtype) / steps, torch.tensor(mask_list, device=device, dtype=torch.bool)
