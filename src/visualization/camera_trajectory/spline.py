import torch
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

def interpolate_extrinsics_batched(key_extrinsics: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Batched interpolation of camera extrinsics.

    Args:
        key_extrinsics (torch.Tensor): [B, K, 4, 4] batch of B sequences with K extrinsics each.
        num_samples (int): Number of samples to interpolate per batch (must be >= K).

    Returns:
        torch.Tensor: [B, num_samples, 4, 4] interpolated extrinsics per batch item.
    """
    B, K, _, _ = key_extrinsics.shape
    assert num_samples >= K, "num_samples must be >= number of keypoints"

    # Prepare output container
    interpolated_all = []

    for b in range(B):
        key = key_extrinsics[b]  # [K, 4, 4]
        R_key = key[:K, :3, :3].cpu().numpy()
        t_key = key[:K, :3, 3].cpu().numpy()
        rot_key = R.from_matrix(R_key)

        segment_count = K - 1
        samples_per_segment = [1] * segment_count
        remaining = num_samples - K
        for i in range(remaining):
            samples_per_segment[i % segment_count] += 1

        slerp = Slerp(np.linspace(0, 1, K), rot_key)

        interpolated = []
        accumulated = 0
        for i in range(segment_count):
            r0, r1 = rot_key[i], rot_key[i + 1]
            t0, t1 = t_key[i], t_key[i + 1]

            n = samples_per_segment[i]
            t_vals = np.linspace(0, 1, n + 2)[1:-1]  # exclude endpoints
            global_t_vals = np.linspace(i, i + 1, n + 2)[1:-1] / (K - 1)

            interp_rots = slerp(global_t_vals)
            interp_trans = (1 - t_vals[:, None]) * t0 + t_vals[:, None] * t1

            # First time: include first keypoint
            if i == 0:
                interpolated.append((r0, t0))

            for r, t in zip(interp_rots, interp_trans):
                interpolated.append((r, t))

            interpolated.append((r1, t1))

        # Convert back to torch tensor
        extrinsics_b = []
        for r, t in interpolated:
            mat = torch.eye(4)
            mat[:3, :3] = torch.from_numpy(r.as_matrix()).float()
            mat[:3, 3] = torch.from_numpy(t).float()
            extrinsics_b.append(mat)

        interpolated_all.append(torch.stack(extrinsics_b, dim=0))

    return torch.stack(interpolated_all, dim=0)  # [B, num_samples, 4, 4]
