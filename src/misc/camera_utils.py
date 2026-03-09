

import torch
import numpy as np

from PIL import Image
from io import BytesIO
from torch import Tensor
from jaxtyping import Float
from typing import Optional, Union
from einops import rearrange, repeat
from ..geometry.projection import get_world_rays, sample_image_grid


def absolute_to_relative_camera(
    tform: Float[Tensor, "batch v 4 4"],
    index: Tensor
):
    _, v, *_ = tform.shape
    ref_tform = tform[index, ...][:, None]
    ref_tform = ref_tform.expand(-1, v, -1, -1) 

    tform = torch.linalg.inv(ref_tform) @ tform
    
    return tform


def generate_image_rays(
    shape,
    extrinsics: Float[Tensor, "batch view 4 4"],
    intrinsics: Float[Tensor, "batch view 3 3"],
    normalize_direction: bool=True
) -> tuple[
    Float[Tensor, "batch view ray 2"],  # xy
    Float[Tensor, "batch view ray 3"],  # origins
    Float[Tensor, "batch view ray 3"],  # directions
]:
    """Generate the rays along which Gaussians are defined. For now, these rays are
    simply arranged in a grid.
    """
    h, w = shape
    b, v, *_ = extrinsics.shape 
    device, dtype = extrinsics.device, extrinsics.dtype
    xy, _ = sample_image_grid((h, w), device=device, dtype=dtype)
    origins, directions = get_world_rays(
        rearrange(xy, "h w xy -> (h w) xy"),
        rearrange(extrinsics, "b v i j -> b v () i j"),
        rearrange(intrinsics, "b v i j -> b v () i j"),
    )
    if normalize_direction:
        directions = directions / (directions.norm(dim=-1, keepdim=True) + 1e-6)

    return (
        repeat(xy, "h w xy -> b v (h w) xy", b=b, v=v), 
        rearrange(origins, "b v (h w) c -> b v c h w", h=h, w=w), 
        rearrange(directions, "b v (h w) c -> b v c h w", h=h, w=w)
    )


def ray_encode(shape, extr, intr, plucker: bool=False, scale: Union[tuple[float, float], list[float, float]]=(1.0, 1.0), normalize_direction: bool=True, switch_ray_order: bool=False):

    intr[..., 0, 0] *= scale[0]
    intr[..., 1, 1] *= scale[1]

    _, origins, directions = generate_image_rays(shape, extr, intr, normalize_direction)


    if plucker:
        origins = torch.cross(origins, directions, dim=2)

    if switch_ray_order: # backward compatibility for legacy experiments
        ray_encodings = torch.concat([origins, directions], dim=2)   
    else:
            
        ray_encodings = torch.concat([directions, origins], dim=2)   
    
    return ray_encodings



def rotation_log(R):
    """
    Compute the logarithm (axis-angle vector) of a batch of SO(3) rotation matrices.
    Args:
        R: (..., 3, 3) batch of rotation matrices
    Returns:
        (..., 3) axis-angle vectors
    """
    cos_theta = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1 + 1e-6, 1 - 1e-6)  # numerical stability
    theta = torch.acos(cos_theta)

    skew = 0.5 * (R - R.transpose(-2, -1))
    axis = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)
    axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-9)

    return axis * theta.unsqueeze(-1)

def pose_distance(pose1, pose2, w_pos=1.0, w_rot=1.0):
    """
    Computes distance matrix between two sets of poses in SE(3).
    pose1, pose2: (N, 4, 4)
    Returns: (N, N) matrix of distances
    """
    t1 = pose1[:, :3, 3]  # (N, 3)
    t2 = pose2[:, :3, 3]  # (N, 3)

    pos_dist = torch.cdist(t1, t2, p=2) ** 2  # (N, N)

    R1 = pose1[:, :3, :3]  # (N, 3, 3)
    R2 = pose2[:, :3, :3]  # (N, 3, 3)

    N = R1.shape[0]
    rel_rot = torch.matmul(R1.unsqueeze(1).transpose(-2, -1), R2.unsqueeze(0))  # (N, N, 3, 3)
    log_rot = rotation_log(rel_rot.view(-1, 3, 3)).view(N, N, 3)
    rot_dist = (log_rot ** 2).sum(dim=-1)  # (N, N)

    return w_pos * pos_dist + w_rot * rot_dist

def fps_from_pose(pose_matrices, n_samples, w_pos=1.0, w_rot=1.0):
    """
    Args:
        pose_matrices: (N, 4, 4) tensor of SE(3) matrices
        n_samples: number of poses to sample
    Returns:
        sorted_indices: (n_samples,) sorted indices of sampled poses
    """
    N = pose_matrices.shape[0]
    device = pose_matrices.device

    dmat = pose_distance(pose_matrices, pose_matrices, w_pos, w_rot)  # (N, N)
    selected = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)

    farthest = torch.randint(0, N, size=(1,), dtype=torch.long).item()

    for i in range(n_samples):
        selected[i] = farthest
        dist_to_farthest = dmat[farthest]
        distances = torch.minimum(distances, dist_to_farthest)
        farthest = torch.argmax(distances).item()

    return torch.sort(selected)[0]

def convert_poses(poses):
    b = poses.shape[0]
    intr = torch.eye(3).repeat(b, 1, 1)
    fx, fy, cx, cy = poses[:, :4].T
    intr[:, 0, 0] = fx
    intr[:, 1, 1] = fy
    intr[:, 0, 2] = cx
    intr[:, 1, 2] = cy
    w2c = torch.eye(4).repeat(b, 1, 1)
    w2c[:, :3] = poses[:, 6:].reshape(b, 3, 4)
    
    return w2c.inverse(), intr

def convert_images(images, preprocess_fn):
    return torch.stack([preprocess_fn(Image.open(BytesIO(im.numpy().tobytes()))) for im in images])


def preprocess_extrinsics(extrinsics, index: Optional[int]=None):
        
        
    b, v, *_ = extrinsics.shape
    device = extrinsics.device

    
    rel_indices_mask = torch.zeros(size=(b, v), device=device)

    rel_indices = torch.tensor(index, device=device).reshape(1, 1).repeat(b, 1)

    rel_indices_mask.scatter_(1, rel_indices, 1.0)
    rel_indices_mask = rel_indices_mask.bool()
    # Transform from absolute to relative poses
    return absolute_to_relative_camera(extrinsics.float(), index=rel_indices_mask).to(device)

def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")

def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return images, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    return center_crop(images, intrinsics, shape)

def reflect_extrinsics(
    extrinsics: Float[Tensor, "*batch 4 4"],
) -> Float[Tensor, "*batch 4 4"]:
    reflect = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    reflect[0, 0] = -1
    return reflect @ extrinsics @ reflect


def reflect_views(image, extrinsics):
    return image.flip(-1), reflect_extrinsics(extrinsics)