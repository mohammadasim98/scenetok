import torch
from typing import Optional
from einops import rearrange, repeat
from .camera_utils import absolute_to_relative_camera

def split_concatenate(inputs, num_views: int, num_set: int):
    _dict = {}
    
    for key, value in inputs.items():
        
        _dict[key] = rearrange(value, "b (num_set num_views) ... -> (b  num_set) num_views ...", num_set=num_set)
    
    return _dict

def sample_arbitrary_views(inputs, nchunks, num_views):

    _dict = {}
    
    for key, value in inputs.items():
        
        _dict[key] = rearrange(value[:, :nchunks * num_views], "b (nchunks v) ... -> (b  nchunks) v ...", nchunks=nchunks)
    
    return _dict


def repeat_batch(inputs, num_set: int):

    _dict = {}
    
    for key, value in inputs.items():
        
        _dict[key] = repeat(value, "b ... -> (b num_set) ...", num_set=num_set).clone()
    
    return _dict  

def batch_interpolate(inputs, nsamples: int):
    
    _dict = {}
    
    for key, value in inputs.items():
        interpolant = []
        for t in range(nsamples):
            t_i = t / (nsamples - 1) 
            interpolant.append((1-t_i) * value[0] + t_i * value[1])
        _dict[key] = torch.stack(interpolant)
    
    return _dict  

def sequence_window(inputs, start: int, end: bool=False):

    _dict = {}
    
    for key, value in inputs.items():
        
        _dict[key] = value[:, start:end, ...]
    
    return _dict     



def sequence_reverse(inputs):
    _dict = {}
    
    for key, value in inputs.items():
        
        _dict[key] = torch.flip(value, dims=(1,))
    
    return _dict

def sequence_concatenate(inputs1, inputs2):
    _dict = {}
    
    for key, value in inputs1.items():
        
        _dict[key] = torch.concat([value, inputs2[key]], dim=1)
    
    return _dict

def sequence_index(inputs, indices):

    _dict = {}
    
    for key, value in inputs.items():
        
        _dict[key] = value[:, indices, ...]
    
    return _dict

def sequence_downsample(inputs, gaps):
    _dict = {}
    
    for key, value in inputs.items():
        
        _dict[key] = value[:, ::gaps, ...]
    
    return _dict

def sequence_limit(inputs, num):
    _dict = {}
    
    for key, value in inputs.items():
        
        _dict[key] = value[:, :num, ...]
    
    return _dict

def batch_expand(inputs):
    _dict = {}
    
    for key, value in inputs.items():
        
        _dict[key] = value[None]
    
    return _dict

def batch_cast(inputs, device):
    _dict = {}
    
    for key, value in inputs.items():
        
        _dict[key] = value.to(device)
    
    return _dict

def preprocess_batch(batch, index: Optional[int]=None):
        
        
    b, v_c, *_ = batch["context"]["extrinsics"].shape
    device = batch["context"]["extrinsics"].device
    v_t = 0
    if "target" in batch.keys():
        b, v_t, *_ = batch["target"]["extrinsics"].shape
    
    rel_indices_mask = torch.zeros(size=(b, v_c+v_t), device=device)

    if index is not None:
        rel_indices = torch.randint(index, index+1, size=(b, 1), device=device)
    else:
        rel_indices = torch.randint(0, v_c, size=(b, 1), device=device)
    
    rel_indices_mask.scatter_(1, rel_indices, 1.0)
    rel_indices_mask = rel_indices_mask.bool()
    # Transform from absolute to relative poses
    if "target" in batch.keys():
        rel_extrinsics = absolute_to_relative_camera(torch.concat([batch["context"]["extrinsics"], batch["target"]["extrinsics"]], dim=1), index=rel_indices_mask).float()
        batch["target"]["extrinsics"] = rel_extrinsics[:, v_c:, ...]
        batch["context"]["extrinsics"] = rel_extrinsics[:, :v_c, ...]
    else:
        batch["context"]["extrinsics"] = absolute_to_relative_camera(batch["context"]["extrinsics"], index=rel_indices_mask).float()
        
    return batch


def preprocess_transfer_batch(extrinsics, index: Optional[int]=None):
        

    b, v, *_ = extrinsics.shape
    device = extrinsics.device

    rel_indices_mask = torch.zeros(size=(b, v), device=device)

    if index is not None:
        rel_indices = torch.randint(index, index+1, size=(b, 1), device=device)
    else:
        rel_indices = torch.randint(0, v, size=(b, 1), device=device)
    
    rel_indices_mask.scatter_(1, rel_indices, 1.0)
    rel_indices_mask = rel_indices_mask.bool()

    extrinsics = absolute_to_relative_camera(extrinsics, index=rel_indices_mask).float()
        
    return extrinsics