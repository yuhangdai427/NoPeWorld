"""A candidate's camera as one Plucker ray per DiT token (float64, then float32), for world models without a frozen per-token camera encoder.

    poses  [N, F, 4, 4]  OpenCV camera-to-world relative to the chunk's state camera, metres
    K      (fx, fy, cx, cy) in pixels of a (height, width) picture
    grid   (rows, cols) of DiT tokens over that picture
    ->     [N, F, rows * cols, 6]  moment o x d and unit direction d
"""
from __future__ import annotations

import numpy as np
import torch


def token_rays(K, height, width, grid):
    """[rows * cols, 3] unit rays of the token centres in the camera frame (x right, y down, z forward)."""
    fx, fy, cx, cy = K
    rows, cols = grid
    u = (np.arange(cols) + 0.5) * width / cols
    v = (np.arange(rows) + 0.5) * height / rows
    uu, vv = np.meshgrid(u, v)
    rays = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], -1).reshape(-1, 3)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def plucker_field(poses, K, height, width, grid, *, device=None):
    poses = np.asarray(poses, np.float64)
    if poses.ndim != 4 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"poses must be [N, F, 4, 4], got {poses.shape}")
    rays = token_rays(K, height, width, grid)
    d = np.einsum("nfij,tj->nfti", poses[..., :3, :3], rays)
    o = np.broadcast_to(poses[..., None, :3, 3], d.shape)
    field = np.concatenate([np.cross(o, d), d], axis=-1)
    return torch.as_tensor(field.astype(np.float32), device=device)


def key_path_poses(rows, step_m):
    """[N, F, 4, 4] poses relative to the state camera at the end of each latent of WorldPlay2 key rows [N, F, 6]
    (pitch, yaw per latent in degrees, forward, lateral, perspective, space). Each latent moves step_m in its key direction at the
    heading half way through its yaw; a turn to the right is positive; the camera stays level."""
    rows = np.asarray(rows, np.float64)
    if rows.ndim != 3 or rows.shape[-1] != 6:
        raise ValueError(f"rows must be [N, F, 6], got {rows.shape}")
    if (rows[..., 0] != 0).any():
        raise ValueError("a row with a pitch: this path is level")
    n, frames = rows.shape[:2]
    poses = np.tile(np.eye(4), (n, frames, 1, 1))
    length = np.hypot(rows[..., 2], rows[..., 3])
    forward = np.divide(rows[..., 2], length, out=np.zeros_like(length), where=length > 0)
    lateral = np.divide(rows[..., 3], length, out=np.zeros_like(length), where=length > 0)
    x, z, heading = np.zeros(n), np.zeros(n), np.zeros(n)
    for f in range(frames):
        yaw = np.radians(rows[:, f, 1])
        mid = heading + 0.5 * yaw
        x = x + step_m * (forward[:, f] * np.sin(mid) + lateral[:, f] * np.cos(mid))
        z = z + step_m * (forward[:, f] * np.cos(mid) - lateral[:, f] * np.sin(mid))
        heading = heading + yaw
        c, s = np.cos(heading), np.sin(heading)
        poses[:, f, 0, 0], poses[:, f, 0, 2], poses[:, f, 2, 0], poses[:, f, 2, 2] = c, s, -s, c
        poses[:, f, 0, 3], poses[:, f, 2, 3] = x, z
    return poses


def scale_intrinsics_cover(width_src, height_src, hfov_deg, height, width):
    """Intrinsics after 'resize to cover, then centre crop' to (height, width) of a source with this horizontal FOV."""
    f_src = 0.5 * width_src / np.tan(np.radians(hfov_deg) / 2)
    scale = max(height / height_src, width / width_src)
    return (f_src * scale, f_src * scale, width / 2.0, height / 2.0)
