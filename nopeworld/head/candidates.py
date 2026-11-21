"""The 34 candidates, their poses at generation time, the intent fit A, and WorldPlay2's key compiler (numpy only).

A candidate is a ground displacement over one chunk at angle delta to the command (+ = right) with length ratio rho.
Directions follow Unreal's yaw sense: + turns clockwise seen from above.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch

_ELLIPSE_DELTA = [8.8, 17.58, 26.4, 35.24, 44.13, 52.84, 60.85, 67.76, 73.66, 78.83, 83.53, 88.0]
_ELLIPSE_RHO = [0.9999, 0.9989, 0.994, 0.9798, 0.9474, 0.8851, 0.7873, 0.6621, 0.5215, 0.3735, 0.2224, 0.0697]
NAMES = (["KEEP", "STOP", "SLOW750", "SLOW500", "SLOW250", "SLOW125", "HLR30", "HLR60"]
         + [f"ELR{d:04.1f}" for d in (8.8, 17.6, 26.4, 35.2, 44.1, 52.8, 60.9, 67.8, 73.7, 78.8, 83.5, 88.0)]
         + ["HLL30", "HLL60"]
         + [f"ELL{d:04.1f}" for d in (8.8, 17.6, 26.4, 35.2, 44.1, 52.8, 60.9, 67.8, 73.7, 78.8, 83.5, 88.0)])
DELTA = [0.0] * 6 + [30.0, 60.0] + _ELLIPSE_DELTA + [-30.0, -60.0] + [-d for d in _ELLIPSE_DELTA]
RHO = [1.0, 0.0, 0.75, 0.5, 0.25, 0.125, 0.433, 0.25] + _ELLIPSE_RHO + [0.433, 0.25] + _ELLIPSE_RHO
KEEP, STOP = 0, 1

SIGMA_A = 0.5
W_LATERAL = 0.5

KEY_SIGNS = {'': (0, 0), 'W': (1, 0), 'A': (0, -1), 'D': (0, 1), 'WA': (1, -1), 'WD': (1, 1)}


def intent_fit(delta_deg, rho, *, w: float = W_LATERAL, sigma: float = SIGMA_A) -> torch.Tensor:
    """A = exp(-e^2 / (2 sigma^2)) with e^2 = (rho cos delta - 1)^2 + w (rho sin delta)^2."""
    delta = torch.deg2rad(torch.as_tensor(delta_deg, dtype=torch.float32))
    rho = torch.as_tensor(rho, dtype=torch.float32)
    e2 = (rho * torch.cos(delta) - 1) ** 2 + w * (rho * torch.sin(delta)) ** 2
    return torch.exp(-e2 / (2 * sigma ** 2))


def candidate_world_poses(poses, plen, *, first, delta=DELTA, rho=RHO):
    """[N, F + 1, 4, 4] camera-to-world poses of one chunk's candidates.

    poses: [F + 1, 4, 4] commanded poses at the chunk's state frame and the end of each latent; plen: [F + 1] commanded ground
    path length from the first of them; first: True for chunk 0. Candidate i moves rho_i Rot(-delta_i) d_u on the ground, spread like
    the command, and keeps the commanded rotations and heights; KEEP (index 0) is the command itself."""
    poses = np.asarray(poses, np.float64)
    plen = np.asarray(plen, np.float64)
    d_u = poses[-1, :2, 3] - poses[0, :2, 3]
    n = len(poses)
    w = plen / plen[-1] if plen[-1] > 1e-9 else (np.concatenate([[0.0], np.arange(n - 1) / (n - 2)]) if first
                                                 else np.arange(n) / (n - 1))
    out = []
    for i, (dl, r) in enumerate(zip(delta, rho)):
        T = poses.copy()
        if i:
            a = np.deg2rad(dl)
            d = r * np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]) @ d_u
            T[:, :2, 3] = poses[0, :2, 3] + w[:, None] * d[None]
        out.append(T)
    return np.array(out)


def path_length(poses, fs):
    """Commanded ground path length (metres) from frame fs[0] to each frame of fs, summed frame by frame."""
    steps = np.linalg.norm(np.diff(np.asarray(poses)[:, :2, 3], axis=0), axis=1)
    return np.array([steps[fs[0]:f].sum() for f in fs])


def resample_poses(poses, t_src, t_dst):
    """Camera-to-world poses at times t_dst from poses at sorted times t_src (linear translation, slerp rotation); times past the
    end continue the last step's motion."""
    poses = np.asarray(poses, np.float64)
    t_src, t_dst = np.asarray(t_src, np.float64), np.asarray(t_dst, np.float64)
    out = np.tile(np.eye(4), (len(t_dst), 1, 1))
    for j, t in enumerate(t_dst):
        i = int(np.clip(np.searchsorted(t_src, t, side="right") - 1, 0, len(t_src) - 2))
        a = (t - t_src[i]) / (t_src[i + 1] - t_src[i])
        relative = poses[i, :3, :3].T @ poses[i + 1, :3, :3]
        angle = np.arccos(np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0))
        if angle < 1e-9:
            R = poses[i, :3, :3] @ (np.eye(3) + a * (relative - np.eye(3)))
        else:
            axis = np.array([relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0],
                             relative[1, 0] - relative[0, 1]]) / (2 * np.sin(angle))
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            R = poses[i, :3, :3] @ (np.eye(3) + np.sin(a * angle) * K + (1 - np.cos(a * angle)) * K @ K)
        out[j, :3, :3] = R
        out[j, :3, 3] = (1 - a) * poses[i, :3, 3] + a * poses[i + 1, :3, 3]
    return out


def rotate(u, deg):
    """A world xy vector turned by `deg` in Unreal's yaw sense."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    u = np.asarray(u, float)
    return np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]])


def unit(key):
    v = np.array(key, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def body_to_world(yaw_deg):
    """[2, 2] from (forward, right) to Unreal xy."""
    fwd = rotate([1.0, 0.0], yaw_deg)
    return np.stack([fwd, rotate(fwd, 90.0)], 1)


def compile_keys(target, yaws, step, *, moving):
    """WorldPlay2: the key sequence for one chunk, per latent (forward, lateral), nearest in end point to `target` (world xy).

    yaws: heading at each moving latent (deg), step: metres per latent, moving: latents that can move. Moving keys first, the one
    nearest to the target's direction first; among equally near sequences the one with fewer moves wins. Returns (keys, end point)."""
    mean = body_to_world(float(np.mean(yaws))) * step
    target_body = np.linalg.solve(mean, np.asarray(target, float))
    side = 1 if target_body[1] >= 0 else -1
    options = [(1, 0), (1, side), (0, side), (0, 0)]
    units = np.array([unit(k) for k in options])
    best, best_cost = None, np.inf
    for combo in itertools.combinations_with_replacement(range(len(options)), moving):
        end = units[list(combo)].sum(0)
        cost = float(np.sum((end - target_body) ** 2)) + 1e-6 * sum(i != 3 for i in combo)
        if cost < best_cost - 1e-12:
            best, best_cost = combo, cost
    direction = target_body / max(np.linalg.norm(target_body), 1e-9)
    order = sorted(best, key=lambda i: (i == 3, -float(units[i] @ direction), i))
    keys = [options[i] for i in order]
    end = np.zeros(2)
    for key, yaw in zip(keys, yaws):
        end += body_to_world(yaw) @ unit(key) * step
    return keys, end


def delta_rho(disp, d_u):
    """Angle to the intent (deg, + = right) and length ratio."""
    U = float(np.linalg.norm(d_u))
    u_hat = np.asarray(d_u, float) / max(U, 1e-9)
    n_hat = rotate(u_hat, 90.0)
    a, b = float(np.asarray(disp) @ u_hat), float(np.asarray(disp) @ n_hat)
    return (math.degrees(math.atan2(b, a)) if abs(a) + abs(b) > 1e-9 else 0.0), math.hypot(a, b) / max(U, 1e-9)


def action_rows(keys, yaws, perspective=1.0):
    """WorldPlay2 action rows [pitch, yaw, forward, lateral, perspective, space] for one chunk (perspective 1 first person, 0 third)."""
    return [[0.0, round(float(y), 4), float(f), float(l), float(perspective), 0.0] for (f, l), y in zip(keys, yaws)]


def runtime_key_candidates(command, yaw_steps, *, first, perspective=1.0):
    """WorldPlay2: the compiled candidates of one chunk from the commanded keys and yaw per latent, in a frame fixed at the chunk's start.

    Returns dict(rows [N, F, 6], delta [N], rho [N], mask [N], length) with KEEP first; KEEP's rows are the command."""
    moving = len(command) - (1 if first else 0)
    headings = []
    heading = 0.0
    for y in yaw_steps:
        headings.append(heading + 0.5 * y)
        heading += y
    keys_cmd, yaw_cmd = list(command[-moving:]), headings[-moving:]
    d_u = sum(body_to_world(y) @ unit(k) for k, y in zip(keys_cmd, yaw_cmd))
    U = float(np.linalg.norm(d_u))
    u_hat = d_u / max(U, 1e-9)
    rows, deltas, rhos, mask, seen = [], [], [], [], {}
    for i, (dl, r) in enumerate(zip(DELTA, RHO)):
        if i == 0:
            keys, end = keys_cmd, d_u
        else:
            keys, end = compile_keys(r * U * rotate(u_hat, dl), yaw_cmd, 1.0, moving=moving)
        d, rr = delta_rho(end, d_u) if i else (0.0, 1.0)
        rows.append(action_rows(([(0, 0)] if first else []) + list(keys), yaw_steps, perspective))
        deltas.append(round(d, 2))
        rhos.append(round(rr, 4))
        mask.append(seen.setdefault(tuple(keys), i) == i)
    return dict(rows=rows, delta=deltas, rho=rhos, mask=mask, length=U)
