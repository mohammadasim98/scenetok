"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from typing import Optional
import torch
from torch.types import _size
import torch.nn as nn


def freeze_model(model: nn.Module) -> None:
    """Freeze the torch model"""
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def bernoulli_tensor(
    size: _size,
    p: float,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
):
    """
    Generate a tensor of the given size,
    where each element is sampled from a Bernoulli distribution with probability `p`.
    """
    return torch.bernoulli(torch.full(size, p, device=device), generator=generator)

def freeze(m: nn.Module) -> None:
    for param in m.parameters():
        param.requires_grad = False
    m.eval()


def unfreeze(m: nn.Module) -> None:
    for param in m.parameters():
        param.requires_grad = True
    m.train()

def unfreeze_buffer_as_parameter(module: nn.Module, name: str, requires_grad: bool = True):
    """Convert a buffer back into a trainable parameter under the same name."""
    if name not in module._buffers:
        raise ValueError(f"{name} is not a buffer of {module.__class__.__name__}")
    
    # Get tensor from buffer
    tensor = module._buffers[name]
    
    # Remove it from buffers
    del module._buffers[name]
    
    # Wrap as Parameter
    param = nn.Parameter(tensor, requires_grad=requires_grad)
    
    # Register as parameter under the same name
    module.register_parameter(name, param)

def freeze_as_buffer(module: nn.Module, name: str):
    """Convert a parameter to a buffer under the same name."""
    if name not in module._parameters:
        raise ValueError(f"{name} is not a parameter of {module.__class__.__name__}")
    
    # Detach to make sure it's not connected to graph
    tensor = module._parameters[name].detach()
    
    # Remove from parameter dict
    del module._parameters[name]
    
    # Register as buffer with the same name
    module.register_buffer(name, tensor)

def convert_to_buffer(module: torch.nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


def replace_keys_substring(d, key_subst_map):

    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            new_key = k
            for old_substr, new_substr in key_subst_map.items():
                new_key = new_key.replace(old_substr, new_substr)
            new_dict[new_key] = replace_keys_substring(v, key_subst_map)
        return new_dict
    elif isinstance(d, list):
        return [replace_keys_substring(i, key_subst_map) for i in d]
    else:
        return d
    

def pop_state_dict_by_prefix(state_dict, prefix):

    out = {}
    matched = False

    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
            matched = True
            
    return out if matched else state_dict