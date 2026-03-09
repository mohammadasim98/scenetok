import torch
import numpy as np

from typing import Literal
    

def load_data(npz_path: str, key: Literal["latent", "image"]):
    data = np.load(npz_path)
    view = torch.from_numpy(data[key])  # Convert to float32 for use
    extrinsics = torch.from_numpy(data['extrinsics'])
    intrinsics = torch.from_numpy(data['intrinsics'])
    
    return view, extrinsics, intrinsics
