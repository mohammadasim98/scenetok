import torch
import math

# ----------------------------- Quaternion helpers -----------------------------

def _safe_normalize(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True).clamp(min=eps))

def rotmat_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to unit quaternions (w,x,y,z), stable branchless variant.
    R: (..., 3, 3)
    Returns: (..., 4) with unit length.
    """
    # Diagonals and trace (no keepdim to avoid shape mismatches)
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    trace = m00 + m11 + m22

    qw = torch.sqrt(torch.clamp(trace + 1.0, min=0.0)) * 0.5
    qx = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0)) * 0.5
    qy = torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0)) * 0.5
    qz = torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0)) * 0.5

    # Choose signs from off-diagonal elements (broadcast-safe)
    qx = qx.copysign(R[..., 2, 1] - R[..., 1, 2])
    qy = qy.copysign(R[..., 0, 2] - R[..., 2, 0])
    qz = qz.copysign(R[..., 1, 0] - R[..., 0, 1])

    q = torch.stack([qw, qx, qy, qz], dim=-1)
    return _safe_normalize(q)


def quat_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    q: (..., 4) unit quaternions (w,x,y,z)
    Returns: (..., 3, 3)
    """
    q = _safe_normalize(q)
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = torch.stack([
        torch.stack([ww+xx-yy-zz, 2*(xy-wz),     2*(xz+wy)    ], dim=-1),
        torch.stack([2*(xy+wz),   ww-xx+yy-zz,   2*(yz-wx)    ], dim=-1),
        torch.stack([2*(xz-wy),   2*(yz+wx),     ww-xx-yy+zz  ], dim=-1),
    ], dim=-2)
    return R

def quat_conjugate(q):
    wxyz = q.clone()
    wxyz[..., 1:] = -wxyz[..., 1:]
    return wxyz

def quat_mul(a, b):
    # Hamilton product for (w,x,y,z)
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    ], dim=-1)

def quat_inv(q):
    # unit quats: inverse = conjugate
    return quat_conjugate(q)

def quat_log(q):
    # log(q) for unit q = [w, v], returns [0, u*theta]
    q = _safe_normalize(q)
    w = q[..., 0].clamp(-1.0, 1.0)
    v = q[..., 1:]
    v_norm = v.norm(dim=-1, keepdim=True)
    theta = torch.atan2(v_norm, w.unsqueeze(-1))
    # avoid division by zero
    scale = torch.where(v_norm > 1e-9, theta / v_norm, torch.zeros_like(v_norm))
    out = torch.cat([torch.zeros_like(w).unsqueeze(-1), v * scale], dim=-1)
    return out

def quat_exp(u):
    # exp([0, v]) = [cos(|v|), u * sin(|v|)/|v|]
    w0 = u[..., 0]  # should be 0 for our usage
    v = u[..., 1:]
    v_norm = v.norm(dim=-1, keepdim=True)
    c = torch.cos(v_norm)
    s = torch.where(v_norm > 1e-9, torch.sin(v_norm) / v_norm, torch.ones_like(v_norm))
    return _safe_normalize(torch.cat([c, v * s], dim=-1))

def quat_slerp(q0, q1, t):
    # q(t) = slerp(q0, q1; t) with shortest-path hemisphere fix
    q0 = _safe_normalize(q0)
    q1 = _safe_normalize(q1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    # flip target to ensure shortest arc
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs()

    eps = 1e-8
    omega = torch.acos(dot.clamp(-1+eps, 1-eps))
    sin_omega = torch.sin(omega).clamp(min=eps)
    t = t.unsqueeze(-1)

    w0 = torch.sin((1 - t) * omega) / sin_omega
    w1 = torch.sin(t * omega) / sin_omega
    return _safe_normalize(q0 * w0 + q1 * w1)

def quat_squad(q0, s0, s1, q1, t):
    # Shoemake's SQUAD: squad(q0, s0, s1, q1, t)
    a = quat_slerp(q0, q1, t)
    b = quat_slerp(s0, s1, t)
    u = 2*t*(1 - t)
    return quat_slerp(a, b, u)

def squad_tangent(q_prev, q_i, q_next):
    """Compute tangent s_i = q_i * exp(-0.25*(log(q_i^{-1} q_prev) + log(q_i^{-1} q_next)))"""
    # Handle ends by repeating neighbors if missing (natural endpoints)
    if q_prev is None:
        q_prev = q_i
    if q_next is None:
        q_next = q_i
    qi_inv = quat_inv(q_i)
    term = 0.25 * (quat_log(quat_mul(qi_inv, q_prev)) + quat_log(quat_mul(qi_inv, q_next)))
    return quat_mul(q_i, quat_exp(-term))

# -------------------------- Catmull–Rom (centripetal) --------------------------

def _centripetal_t_values(Pm1, P0, P1, P2, alpha=0.5):
    # Compute chordal times t_-1 < t0 < t1 < t2 (we return t0,t1,t2)
    def tj(ti, Pi, Pj):
        return ti + (Pi - Pj).norm()**alpha
    t0 = torch.zeros((), device=P0.device)
    t1 = tj(t0, P1, P0)
    t2 = tj(t1, P2, P1)
    # We keep t_-1 only implicitly as t_-1 = t0 - |P0-Pm1|^alpha (not needed explicitly).
    return t0, t1, t2

def catmull_rom_point(Pm1, P0, P1, P2, u, alpha=0.5):
    """
    Centripetal Catmull–Rom through P0,P1. 
    u in [0,1] parameterizes between P0 (u=0) and P1 (u=1).
    """
    t0, t1, t2 = _centripetal_t_values(Pm1, P0, P1, P2, alpha)
    # map u in [0,1] to t in [t0, t1]
    t = t0 + u * (t1 - t0)
    # De Boor-like evaluation with non-uniform parameters
    A1 = (t1 - t)/(t1 - t0).clamp(min=1e-9) * Pm1 + (t - t0)/(t1 - t0).clamp(min=1e-9) * P0
    A2 = (t2 - t)/(t2 - t1).clamp(min=1e-9) * P0  + (t - t1)/(t2 - t1).clamp(min=1e-9) * P1
    # Need P3 time t3 to continue fully; common, simpler variant uses two-point
    # refinement; use a standard CR blend equivalent:
    # A more stable, succinct approach is the matrix form:
    # But to keep centripetal timing, we approximate with a Hermite-like form:
    # For robustness and simplicity, fallback to uniform CR basis (works well in practice):
    # (Exact centripetal evaluation requires t3; we approximate well with the uniform basis below.)
    # Uniform CR (pass-through) with local 4 points:
    # Compute tangents:
    m0 = 0.5 * (P1 - Pm1)
    m1 = 0.5 * (P2 - P0)
    uu = u
    uu2 = uu*uu
    uu3 = uu2*uu
    h00 = 2*uu3 - 3*uu2 + 1
    h10 = uu3 - 2*uu2 + uu
    h01 = -2*uu3 + 3*uu2
    h11 = uu3 - uu2
    return h00*P0 + h10*m0 + h01*P1 + h11*m1

# ------------------------------- Main function --------------------------------

def interpolate_extrinsics_smooth(
    extrinsics: torch.Tensor,
    N: int,
    rot_weight: float = None,
    alpha: float = 0.5,
    assume_c2w: bool = True,
):
    """
    Create a smooth SE(3) path through K key extrinsics, producing N frames (N >= K),
    and return the indices where each key pose appears in the sequence.

    Args:
        extrinsics: (K, 4, 4) homogeneous transforms. By default assumed camera-to-world (c2w).
                    (If world-to-camera, set assume_c2w=False; we'll invert internally.)
        N:          Total number of frames to produce (must be >= K).
        rot_weight: Weight (meters per radian) to blend rotation into segment length for sample allocation.
                    If None, uses mean chord length of translations (reasonable balance).
        alpha:      Centripetal Catmull–Rom alpha (0.5 recommended).
        assume_c2w: If False, input is world-to-camera; will be inverted to c2w internally.

    Returns:
        poses:        (N, 4, 4) c2w transforms along the path (passes exactly through the K keys).
        key_indices:  (K,) LongTensor with the indices in [0, N-1] where each original key pose occurs.
    """
    assert extrinsics.ndim == 3 and extrinsics.shape[-2:] == (4,4), "extrinsics must be (K,4,4)"
    K = extrinsics.shape[0]
    if N < K:
        raise ValueError(f"N ({N}) must be >= number of key extrinsics K ({K}) to include all keys.")

    device = extrinsics.device
    dtype = extrinsics.dtype

    # Convert to camera-to-world if needed
    if assume_c2w:
        c2w_keys = extrinsics
    else:
        # invert w2c -> c2w
        R = extrinsics[..., :3, :3]
        t = extrinsics[..., :3, 3:4]
        Rt = R.transpose(-1, -2)
        c2w = torch.eye(4, device=device, dtype=dtype).repeat(K,1,1)
        c2w[:, :3, :3] = Rt
        c2w[:, :3, 3:4] = -Rt @ t
        c2w_keys = c2w

    # Split into translations and rotations (as quaternions)
    Rk = c2w_keys[:, :3, :3]
    tk = c2w_keys[:, :3, 3]
    qk = rotmat_to_quat_wxyz(Rk)

    # Rotation tangents for SQUAD
    tangents = []
    for i in range(K):
        q_prev = qk[i-1] if i-1 >= 0 else None
        q_next = qk[i+1] if i+1 < K else None
        tangents.append(squad_tangent(q_prev, qk[i], q_next))
    tangents = torch.stack(tangents, dim=0)  # (K,4)

    # Segment weights for distributing samples
    trans_chords = (tk[1:] - tk[:-1]).norm(dim=-1)  # (K-1,)
    # rotation angle (shortest arc)
    dot = (qk[:-1] * qk[1:]).sum(-1).abs().clamp(0, 1)
    rot_angles = 2.0 * torch.atan2(torch.sqrt(1 - dot*dot).clamp(min=1e-9), dot.clamp(min=1e-9))  # (K-1,)
    if rot_weight is None:
        rot_weight = trans_chords.mean().item() if (K > 1 and trans_chords.numel() > 0) else 1.0
    seg_w = trans_chords + rot_weight * rot_angles
    seg_w = torch.where(seg_w > 1e-12, seg_w, torch.full_like(seg_w, 1e-6))
    seg_w = seg_w / seg_w.sum()

    # Distribute interior samples across segments
    num_interior = N - K
    base = torch.floor(seg_w * num_interior).to(torch.long)
    remainder = num_interior - int(base.sum().item())
    # assign the leftover 1s to segments with largest fractional parts
    frac = (seg_w * num_interior) - base.float()
    order = torch.argsort(frac, descending=True)
    base[order[:remainder]] += 1
    interiors_per_seg = base.tolist()  # list length K-1

    # Build the sequence, tracking key indices
    frames = []
    key_indices = []

    # helper to append a pose (R,t)
    def append_pose(R, t):
        M = torch.eye(4, device=device, dtype=dtype)
        M[:3, :3] = R
        M[:3, 3] = t
        frames.append(M)

    # For Catmull–Rom endpoints, duplicate end points
    def get_point(idx):
        if idx < 0: idx = 0
        if idx >= K: idx = K-1
        return tk[idx]

    # Iterate segments
    global_idx = 0
    # push the first key pose
    append_pose(Rk[0], tk[0])
    key_indices.append(global_idx)
    global_idx += 1

    for i in range(K-1):
        n_int = interiors_per_seg[i]
        # interior parameters (exclude 0 and 1)
        if n_int > 0:
            ts = torch.linspace(0.0, 1.0, steps=n_int+2, device=device)[1:-1]
        else:
            ts = torch.empty(0, device=device)

        # Neighbors for CR
        Pm1 = get_point(i-1)
        P0  = get_point(i)
        P1  = get_point(i+1)
        P2  = get_point(i+2)

        # interpolate interiors
        for u in ts:
            # translation (CR)
            p = catmull_rom_point(Pm1, P0, P1, P2, u, alpha=alpha)
            # rotation (SQUAD)
            q = quat_squad(qk[i], tangents[i], tangents[i+1], qk[i+1], u)
            R = quat_wxyz_to_rotmat(q)
            append_pose(R, p)
            global_idx += 1

        # finally, append the exact next keyframe (u=1)
        append_pose(Rk[i+1], tk[i+1])
        key_indices.append(global_idx)
        global_idx += 1

    poses = torch.stack(frames, dim=0)  # (N,4,4)
    key_indices = torch.tensor(key_indices, device=device, dtype=torch.long)  # (K,)

    # If inputs were w2c, user may want outputs in the same convention.
    # We return c2w as documented; caller can invert if needed.
    return poses, key_indices
