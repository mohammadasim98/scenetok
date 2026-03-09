import torch
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt

def loop_trajectory_full_spin(extrinsics: torch.Tensor, extra_frames: int = 36, spin_axis='y'):
    """
    Complete a trajectory into a loop that also spins full 360° smoothly along the spin_axis.
    
    Args:
        extrinsics: (N,4,4) camera-to-world matrices
        extra_frames: number of interpolated frames
        spin_axis: 'x', 'y', or 'z'
        
    Returns:
        (N+extra_frames, 4,4) tensor of extrinsics
    """
    device, dtype = extrinsics.device, extrinsics.dtype
    N = extrinsics.shape[0]

    # Extract translation and rotation
    translations = extrinsics[:, :3, 3].cpu().numpy()
    rotations = R.from_matrix(extrinsics[:, :3, :3].cpu().numpy())

    # Spin axis vector in world coordinates
    axis_dict = {'x': np.array([1,0,0]), 'y': np.array([0,1,0]), 'z': np.array([0,0,1])}
    axis = axis_dict[spin_axis]

    # Interpolation parameter
    t_vals = np.linspace(0,1,extra_frames+1)[1:]  # skip 0

    # Linear interpolate translation from last → first
    trans_last, trans_first = translations[-1], translations[0]
    interp_trans = np.array([(1-t)*trans_last + t*trans_first for t in t_vals])

    # For rotation: take last rotation as starting point
    rot_last = rotations[-1]

    # Create rotation of 0 → 2π around spin axis
    spins = R.from_rotvec(np.outer(t_vals*2*np.pi, axis))  # shape (extra_frames,)
    
    # Compose last rotation with spin
    interp_rots = rot_last * spins

    # Build extrinsics
    new_extrinsics = []
    for Rmat, tvec in zip(interp_rots.as_matrix(), interp_trans):
        ext = torch.eye(4, device=device, dtype=dtype)
        ext[:3, :3] = torch.tensor(Rmat, device=device, dtype=dtype)
        ext[:3, 3] = torch.tensor(tvec, device=device, dtype=dtype)
        new_extrinsics.append(ext)

    return torch.cat([extrinsics, torch.stack(new_extrinsics)], dim=0)
def loop_trajectory_360_fixed(extrinsics: torch.Tensor, extra_frames: int = 30, spin_axis: str = "y"):
    """
    Complete a trajectory into a smooth 360° loop without backward jump.
    """
    device, dtype = extrinsics.device, extrinsics.dtype
    N = extrinsics.shape[0]

    translations = extrinsics[:, :3, 3].cpu().numpy()  # (N,3)
    rotations = R.from_matrix(extrinsics[:, :3, :3].cpu().numpy())
    eulers = rotations.as_euler("xyz", degrees=False)

    axis_idx = {"x": 0, "y": 1, "z": 2}[spin_axis]

    # Unwrap spin axis to avoid sudden jumps
    eulers[:, axis_idx] = np.unwrap(eulers[:, axis_idx])

    # Current last angle on spin axis
    last_angle = eulers[-1, axis_idx]
    # Define target angle for a full rotation
    target_angle = last_angle + 2 * np.pi
    # Replace last frame's angle for interpolation
    interp_start = last_angle
    interp_end = target_angle

    # Interpolation parameter
    t_vals = np.linspace(0, 1, extra_frames + 1)[1:]  # exclude 0 (already last frame)

    # Interpolate translations
    trans_last, trans_first = translations[-1], translations[0]
    interp_trans = np.array([
        (1 - t) * trans_last + t * trans_first for t in t_vals
    ])

    # Interpolate rotations
    interp_eulers = np.tile(eulers[-1], (extra_frames, 1))  # copy last eulers
    interp_eulers[:, axis_idx] = interp_start + t_vals * (interp_end - interp_start)

    # Other axes: interpolate linearly from last → first
    for idx in range(3):
        if idx != axis_idx:
            interp_eulers[:, idx] = eulers[-1, idx] * (1 - t_vals) + eulers[0, idx] * t_vals

    # Convert back to rotation matrices
    interp_rots = R.from_euler("xyz", interp_eulers, degrees=False).as_matrix()

    # Build extrinsics
    new_extrinsics = []
    for Rmat, tvec in zip(interp_rots, interp_trans):
        ext = torch.eye(4, device=device, dtype=dtype)
        ext[:3, :3] = torch.tensor(Rmat, device=device, dtype=dtype)
        ext[:3, 3] = torch.tensor(tvec, device=device, dtype=dtype)
        new_extrinsics.append(ext)

    return torch.cat([extrinsics, torch.stack(new_extrinsics)], dim=0)


def loop_trajectory(extrinsics: torch.Tensor, extra_frames: int = 20):
    """
    Given a sequence of extrinsics (camera-to-world), extend it so the trajectory loops
    back to the start smoothly.

    Args:
        extrinsics (torch.Tensor): (N, 4, 4) camera-to-world extrinsics.
        extra_frames (int): number of interpolated frames between last and first.

    Returns:
        torch.Tensor: (N + extra_frames, 4, 4) new looped extrinsics.
    """
    assert extrinsics.ndim == 3 and extrinsics.shape[1:] == (4, 4)
    device, dtype = extrinsics.device, extrinsics.dtype

    # Extract translations and rotations
    translations = extrinsics[:, :3, 3].cpu().numpy()   # (N, 3)
    rotations = extrinsics[:, :3, :3].cpu().numpy()     # (N, 3, 3)

    # --- Rotation interpolation setup ---
    key_times = [0, 1]
    key_rots = R.from_matrix([rotations[-1], rotations[0]])
    slerp = Slerp(key_times, key_rots)

    # Interpolation points (skip exact 0 and 1, since we already have them)
    t_vals = np.linspace(0, 1, extra_frames + 2)[1:-1]

    # Interpolated rotations
    interp_rots = slerp(t_vals).as_matrix()   # (extra_frames, 3, 3)

    # Interpolated translations (simple linear)
    trans_last, trans_first = translations[-1], translations[0]
    interp_trans = np.array([
        (1 - t) * trans_last + t * trans_first for t in t_vals
    ])  # (extra_frames, 3)

    # --- Build new extrinsics ---
    new_extrinsics = []
    for trans, rot in zip(interp_trans, interp_rots):
        ext = torch.eye(4, device=device, dtype=dtype)
        ext[:3, :3] = torch.tensor(rot, device=device, dtype=dtype)
        ext[:3, 3] = torch.tensor(trans, device=device, dtype=dtype)
        new_extrinsics.append(ext)

    # Concatenate: original + interpolated
    all_ext = torch.cat([extrinsics, torch.stack(new_extrinsics, dim=0)], dim=0)
    return all_ext


def loop_trajectory_360(extrinsics: torch.Tensor, extra_frames: int = 30, spin_axis: str = "y"):
    """
    Given a trajectory of extrinsics, complete it into a closed loop while spinning 360°.
    
    Args:
        extrinsics (torch.Tensor): (N, 4, 4) camera-to-world matrices (OpenCV convention).
        extra_frames (int): number of interpolated frames between last and first.
        spin_axis (str): axis to spin around: "x", "y", or "z".
        
    Returns:
        torch.Tensor: (N+extra_frames, 4, 4) looped trajectory.
    """
    device, dtype = extrinsics.device, extrinsics.dtype
    N = extrinsics.shape[0]

    # Extract translations
    translations = extrinsics[:, :3, 3].cpu().numpy()  # (N, 3)

    # Extract rotations as Euler angles (to unwrap spin)
    rotations = R.from_matrix(extrinsics[:, :3, :3].cpu().numpy())  
    eulers = rotations.as_euler("xyz", degrees=False)  # (N, 3)

    # Select spin axis index
    axis_idx = {"x": 0, "y": 1, "z": 2}[spin_axis]

    # Unwrap spin axis to make sure it accumulates (no shortest path)
    eulers[:, axis_idx] = np.unwrap(eulers[:, axis_idx])

    # Force the last spin to be +2π beyond the first (for full rotation)
    eulers[-1, axis_idx] = eulers[0, axis_idx] + 2 * np.pi

    # Interpolation parameter
    t_vals = np.linspace(0, 1, extra_frames + 1)[1:]  # exclude 0 (already last frame)

    # Interpolate translations (linear)
    trans_last, trans_first = translations[-1], translations[0]
    interp_trans = np.array([
        (1 - t) * trans_last + t * trans_first for t in t_vals
    ])  # (extra_frames, 3)

    # Interpolate Euler angles (linear, with unwrapped spin)
    rot_last, rot_first = eulers[-1], eulers[0] + np.array([0,0,0])  # keep continuity
    interp_eulers = np.array([
        (1 - t) * rot_last + t * rot_first for t in t_vals
    ])  # (extra_frames, 3)

    # Convert back to rotation matrices
    interp_rots = R.from_euler("xyz", interp_eulers, degrees=False).as_matrix()

    # Build interpolated extrinsics
    new_extrinsics = []
    for Rmat, tvec in zip(interp_rots, interp_trans):
        ext = torch.eye(4, device=device, dtype=dtype)
        ext[:3, :3] = torch.tensor(Rmat, device=device, dtype=dtype)
        ext[:3, 3] = torch.tensor(tvec, device=device, dtype=dtype)
        new_extrinsics.append(ext)

    return torch.cat([extrinsics, torch.stack(new_extrinsics)], dim=0)


def circular_trajectory(center_extrinsic: torch.Tensor, num_frames: int = 36, radius: float = 2.0):
    """
    Generate a smooth 360° circular camera trajectory around a given extrinsic.
    OpenCV convention: looking along -Z, Y is up, X is right.

    Args:
        center_extrinsic (torch.Tensor): (4,4) reference camera-to-world extrinsic.
        num_frames (int): number of frames in the circular trajectory.
        radius (float): circle radius.

    Returns:
        torch.Tensor: (num_frames, 4, 4) circular extrinsics.
    """
    assert center_extrinsic.shape == (4, 4)
    device, dtype = center_extrinsic.device, center_extrinsic.dtype

    # Get reference position
    center_pos = center_extrinsic[:3, 3]

    # Angles for full circle
    angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)

    extrinsics = []
    for theta in angles:
        # Circle position (XZ plane for top view, keeping Y fixed)
        pos = torch.tensor([
            radius * np.cos(theta),
            center_pos[1].item(),
            radius * np.sin(theta)
        ], device=device, dtype=dtype)

        # Look-at rotation (camera looks at center position)
        forward = (center_pos - pos)
        forward = forward / forward.norm()

        up = torch.tensor([0, 1, 0], device=device, dtype=dtype)
        right = torch.cross(forward, up)
        right = right / right.norm()
        up = torch.cross(right, forward)

        Rcw = torch.stack([right, up, -forward], dim=1)  # OpenCV convention

        # Build extrinsic
        ext = torch.eye(4, device=device, dtype=dtype)
        ext[:3, :3] = Rcw
        ext[:3, 3] = pos
        extrinsics.append(ext)

    return torch.stack(extrinsics, dim=0)

def loop_back_trajectory_curved_radius(orig_extrinsics: torch.Tensor,
                                       extra_frames: int = 36,
                                       spin_axis: str = 'y',
                                       spin: bool = True,
                                       radius: float = 1.0):
    """
    Extend a trajectory into a smooth curved loop with adjustable radius.
    Camera looks along the tangent of the curve with optional 360° spin.

    Args:
        orig_extrinsics: (N,4,4) original trajectory
        extra_frames: number of frames along return path
        spin_axis: axis to spin around ('x','y','z')
        spin: whether to rotate 360° around tangent
        radius: controls how far the return path bends (larger radius = more curved)

    Returns:
        (N + extra_frames, 4, 4) extended trajectory
    """
    device, dtype = orig_extrinsics.device, orig_extrinsics.dtype

    # Extract positions
    positions = orig_extrinsics[:, :3, 3].cpu().numpy()
    N = positions.shape[0]

    start_pos = positions[0]
    end_pos = positions[-1]

    # Tangents
    if N > 1:
        tangent_start = positions[1] - positions[0]
        tangent_end = positions[-1] - positions[-2]
    else:
        tangent_start = np.array([1.0,0,0])
        tangent_end = tangent_start

    tangent_start /= np.linalg.norm(tangent_start)
    tangent_end /= np.linalg.norm(tangent_end)

    # Add perpendicular component for curvature
    line_dir = end_pos - start_pos
    line_dir /= np.linalg.norm(line_dir)

    up = np.array([0,1,0])
    if np.abs(np.dot(up, line_dir)) > 0.9:
        up = np.array([1,0,0])
    bend_dir = np.cross(line_dir, up)
    bend_dir /= np.linalg.norm(bend_dir)

    tangent_start = tangent_start + radius * bend_dir
    tangent_end = tangent_end + radius * bend_dir
    tangent_start /= np.linalg.norm(tangent_start)
    tangent_end /= np.linalg.norm(tangent_end)

    # Cubic Hermite spline
    t_vals = np.linspace(0,1,extra_frames)
    positions_interp = np.zeros((extra_frames,3))
    for i,t in enumerate(t_vals):
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        positions_interp[i] = h00*end_pos + h10*tangent_end + h01*start_pos + h11*tangent_start

    # Compute tangents along curve
    tangents = np.gradient(positions_interp, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]

    # Flip tangent to point along curve direction (last → first)
    tangents = -tangents

    # Optional spin
    axis_dict = {'x': np.array([1,0,0]), 'y': np.array([0,1,0]), 'z': np.array([0,0,1])}
    spin_axis_vec = axis_dict[spin_axis]
    spin_angles = np.linspace(0, 1.4*np.pi, extra_frames) if spin else np.zeros(extra_frames)

    # Build extrinsics
    new_extrinsics = []
    for i in range(extra_frames):
        forward = tangents[i]
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        cam_up = np.cross(right, forward)

        if spin:
            spin_rot = R.from_rotvec(spin_angles[i]*forward).as_matrix()
            Rcw = spin_rot @ np.column_stack([right, cam_up, -forward])
        else:
            Rcw = np.column_stack([right, cam_up, -forward])

        ext = torch.eye(4, device=device, dtype=dtype)
        ext[:3,:3] = torch.tensor(Rcw, device=device, dtype=dtype)
        ext[:3,3] = torch.tensor(positions_interp[i], device=device, dtype=dtype)
        new_extrinsics.append(ext)

    return torch.cat([orig_extrinsics, torch.stack(new_extrinsics)], dim=0)


def fig_to_numpy(fig, close=True):
    """
    Convert a matplotlib figure to a (H, W, 3) uint8 numpy array (RGB).
    """
    fig.canvas.draw()  # Render the figure
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = data.reshape((h, w, 3))
    if close:
        plt.close(fig)
    return img

def plot_camera_extrinsics_top_view(extrinsics: torch.Tensor, invert: bool = False, index: int | None=0):
    """
    Plot camera extrinsics from a top view (XZ plane).
    
    Args:
        extrinsics (torch.Tensor): Tensor of shape (N, 4, 4), OpenCV convention.
        invert (bool): If True, treat input as world-to-camera and invert to get camera-to-world.
    """
    assert extrinsics.ndim == 3 and extrinsics.shape[1:] == (4, 4), \
        "Extrinsics must have shape (N, 4, 4)"
    
    N = extrinsics.shape[0]
    
    # If extrinsics are world-to-camera, invert them
    if invert:
        extrinsics = torch.linalg.inv(extrinsics)
    
    # Extract camera centers (translation part of camera-to-world matrix)
    centers = extrinsics[:, :3, 3].cpu().numpy()
    
    # Extract forward direction (camera's +Z axis in OpenCV convention)
    # In camera-to-world matrix, columns 0,1,2 are the camera's X,Y,Z axes in world coords
    forwards = extrinsics[:, :3, 2].cpu().numpy()
    
    # Normalize forward vectors
    forwards = forwards / np.linalg.norm(forwards, axis=1, keepdims=True)
    
    # Plot top view (XZ plane)
    plt.figure(figsize=(6, 6))
    plt.scatter(centers[:, 0], centers[:, 2], c='blue', label='Camera centers')
    
    # Plot viewing directions as arrows
    scale = 0.2 * np.linalg.norm(centers[:, :2].max(axis=0) - centers[:, :2].min(axis=0))
    for i in range(N):
        if i == index and i is not None:
            plt.arrow(
                centers[i, 0], centers[i, 2],
                forwards[i, 0] * scale, forwards[i, 2] * scale,
                head_width=1.5*scale, head_length=1*scale, fc='g', ec='g', zorder=float('inf')
            )
        else:
            plt.arrow(
                centers[i, 0], centers[i, 2],
                forwards[i, 0] * scale, forwards[i, 2] * scale,
                head_width=0.9*scale, head_length=0.9*scale, fc='r', ec='r'
            )
        # plt.text(centers[i, 0], centers[i, 2], s=f"{i}")
    
    plt.xlabel("X (right)")
    plt.ylabel("Z (forward)")
    # plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title("Camera Extrinsics (Top View)")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    # plt.show()

    # fig = plt.gcf()
    # fig.canvas.draw()

    # # Convert RGBA buffer to numpy array
    # img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # plt.close(fig)
    # return img

import torch
import math

def normalize(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))

def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix (3x3 or ...x3x3) to unit quaternion (...,4) in (w,x,y,z) format.
    """
    batch = R.shape[:-2]
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = m00 + m11 + m22

    q = torch.empty((*batch, 4), dtype=R.dtype, device=R.device)
    # Cases adapted for numerical stability
    cond0 = trace > 0
    s0 = torch.sqrt(trace[cond0] + 1.0) * 2
    q0 = torch.empty((cond0.sum(), 4), dtype=R.dtype, device=R.device)
    q0[..., 0] = 0.25 * s0
    q0[..., 1] = (m21[cond0] - m12[cond0]) / s0
    q0[..., 2] = (m02[cond0] - m20[cond0]) / s0
    q0[..., 3] = (m10[cond0] - m01[cond0]) / s0

    cond1 = (~cond0) & (m00 > m11) & (m00 > m22)
    s1 = torch.sqrt(1.0 + m00[cond1] - m11[cond1] - m22[cond1]) * 2
    q1 = torch.empty((cond1.sum(), 4), dtype=R.dtype, device=R.device)
    q1[..., 0] = (m21[cond1] - m12[cond1]) / s1
    q1[..., 1] = 0.25 * s1
    q1[..., 2] = (m01[cond1] + m10[cond1]) / s1
    q1[..., 3] = (m02[cond1] + m20[cond1]) / s1

    cond2 = (~cond0) & (~cond1) & (m11 > m22)
    s2 = torch.sqrt(1.0 + m11[cond2] - m00[cond2] - m22[cond2]) * 2
    q2 = torch.empty((cond2.sum(), 4), dtype=R.dtype, device=R.device)
    q2[..., 0] = (m02[cond2] - m20[cond2]) / s2
    q2[..., 1] = (m01[cond2] + m10[cond2]) / s2
    q2[..., 2] = 0.25 * s2
    q2[..., 3] = (m12[cond2] + m21[cond2]) / s2

    cond3 = (~cond0) & (~cond1) & (~cond2)
    s3 = torch.sqrt(1.0 + m22[cond3] - m00[cond3] - m11[cond3]) * 2
    q3 = torch.empty((cond3.sum(), 4), dtype=R.dtype, device=R.device)
    q3[..., 0] = (m10[cond3] - m01[cond3]) / s3
    q3[..., 1] = (m02[cond3] + m20[cond3]) / s3
    q3[..., 2] = (m12[cond3] + m21[cond3]) / s3
    q3[..., 3] = 0.25 * s3

    q[cond0] = q0
    q[cond1] = q1
    q[cond2] = q2
    q[cond3] = q3
    return normalize(q)

def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternion (...,4) (w,x,y,z) to rotation matrix (...,3,3).
    """
    q = normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = torch.stack([
        torch.stack([ww + xx - yy - zz, 2*(xy - wz),         2*(xz + wy)], dim=-1),
        torch.stack([2*(xy + wz),         ww - xx + yy - zz, 2*(yz - wx)], dim=-1),
        torch.stack([2*(xz - wy),         2*(yz + wx),       ww - xx - yy + zz], dim=-1),
    ], dim=-2)
    return R

def slerp_quat(q0, q1, t):
    """
    q0, q1: (...,4) unit quats; t: (...,) in [0,1]
    Returns (...,4)
    """
    # Ensure shortest path
    dot = (q0 * q1).sum(dim=-1)
    q1 = torch.where(dot[..., None] < 0, -q1, q1)
    dot = torch.abs(dot)

    EPS = 1e-8
    # If very close, do linear
    mask = dot > (1 - 1e-6)
    out = torch.empty_like(q0)

    # Linear part
    out[mask] = normalize((1 - t[mask][..., None]) * q0[mask] + t[mask][..., None] * q1[mask])

    # Slerp part
    theta = torch.acos(dot[~mask].clamp(-1 + EPS, 1 - EPS))
    sin_theta = torch.sin(theta)
    w0 = torch.sin((1 - t[~mask]) * theta) / sin_theta
    w1 = torch.sin(t[~mask] * theta) / sin_theta
    out[~mask] = (w0[..., None] * q0[~mask] + w1[..., None] * q1[~mask])
    return normalize(out)

def extract_center_and_R(E: torch.Tensor, w2c: bool) -> (torch.Tensor, torch.Tensor):
    """
    E: (4,4) extrinsic. If w2c=True, E maps world->camera (OpenCV/Colmap).
    Returns camera center C (3,) in world coords and the rotation R such that:
      - if w2c: R = R_wc (world->cam)
      - if c2w: R = R_cw (cam->world)
    """
    R = E[:3, :3]
    t = E[:3, 3]
    if w2c:
        C = -R.transpose(0,1) @ t
        return C, R
    else:
        C = t
        return C, R

def assemble_extrinsic(C: torch.Tensor, R: torch.Tensor, w2c: bool) -> torch.Tensor:
    """
    Build (4,4) extrinsic from center C (3,) and rotation R (3,3) matching the same convention.
    """
    E = torch.eye(4, dtype=R.dtype, device=R.device)
    E[:3, :3] = R
    if w2c:
        # t = -R * C
        E[:3, 3] = -R @ C
    else:
        # c2w: t = C
        E[:3, 3] = C
    return E

def bezier_quadratic(p0, p1, pc, t):  # all 3D, t (N,)
    t = t[..., None]
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * pc + t**2 * p1  # (N,3)

def pick_side_normal(chord: torch.Tensor, up_hint: torch.Tensor) -> torch.Tensor:
    """
    Returns a unit vector orthogonal to chord, biased by up_hint.
    Falls back to a different hint if nearly colinear.
    """
    n = torch.cross(chord, up_hint)
    if n.norm() < 1e-8:
        alt = torch.tensor([0.0, 0.0, 1.0], device=chord.device, dtype=chord.dtype)
        if torch.allclose(up_hint, alt):
            alt = torch.tensor([0.0, 1.0, 0.0], device=chord.device, dtype=chord.dtype)
        n = torch.cross(chord, alt)
    return normalize(n)

def generate_trajectory_between_extrinsics(
    E0: torch.Tensor,
    E1: torch.Tensor,
    N: int,
    curvature: float = 0.0,
    w2c: bool = True,
    up_hint: torch.Tensor | None = None,
    max_curve_scale: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a smooth trajectory of N extrinsics connecting E0->E1.
    - Positions follow a quadratic Bézier. curvature in [-1,1]:
        0   -> straight line (shortest path)
        +1  -> maximum bend to one side
        -1  -> maximum bend to the opposite side
      The bend side is defined by cross(chord, up_hint).
    - Rotations follow quaternion SLERP between the start/end rotations.

    Args:
        E0, E1: (4,4) extrinsics. If w2c=True, treat as world->camera (OpenCV/Colmap).
        N: number of samples (>=2).
        curvature: float in [-1,1].
        w2c: convention flag.
        up_hint: (3,) world up direction to define bend side. Default [0,1,0].
        max_curve_scale: maximum control offset as a fraction of chord length.

    Returns:
        traj_E: (N,4,4) extrinsics from start (t=0) to end (t=1)
        traj_centers: (N,3) camera centers in world coordinates
    """
    assert N >= 2, "N must be >= 2"
    device = E0.device
    dtype = E0.dtype
    if up_hint is None:
        up_hint = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

    # Extract centers and rotations
    C0, R0 = extract_center_and_R(E0, w2c=w2c)
    C1, R1 = extract_center_and_R(E1, w2c=w2c)

    # Sample parameter t
    t = torch.linspace(0.0, 1.0, N, device=device, dtype=dtype)

    # Control point for quadratic Bézier
    chord = C1 - C0
    chord_len = chord.norm().item()
    if chord_len < 1e-10:
        # Degenerate: same position — just SLERP rotations in place
        q0 = rotmat_to_quat(R0[None, ...])
        q1 = rotmat_to_quat(R1[None, ...])
        qT = slerp_quat(q0.expand(N, -1), q1.expand(N, -1), t)
        RT = quat_to_rotmat(qT)
        CT = C0.expand(N, -1)
    else:
        mid = 0.5 * (C0 + C1)
        side = pick_side_normal(chord, up_hint)
        # Control magnitude scales with chord length
        m = abs(curvature) * max_curve_scale * chord_len
        pc = mid + (torch.sign(torch.tensor(curvature, device=device, dtype=dtype)) if curvature != 0 else 0.0) * m * side
        # Bezier positions
        CT = bezier_quadratic(C0, C1, pc, t)

        # Rotations: SLERP between endpoints
        q0 = rotmat_to_quat(R0[None, ...])
        q1 = rotmat_to_quat(R1[None, ...])
        qT = slerp_quat(q0.expand(N, -1), q1.expand(N, -1), t)
        RT = quat_to_rotmat(qT)

    # Assemble extrinsics along the curve
    traj_E = torch.stack([assemble_extrinsic(CT[i], RT[i], w2c=w2c) for i in range(N)], dim=0)
    return traj_E, CT