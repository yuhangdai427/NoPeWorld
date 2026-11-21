"""Depth and camera motion of a test video from one multi-view reconstruction (VGGT-Omega by default, DA3 optional), in metres.

The history frames are real renders with known poses: the reconstruction is expressed in the hand-off camera's frame (OpenCV: x right,
y down, z forward) and scaled so that the history's camera path has its true length; nothing after the hand-off uses ground truth.
Neither model is shipped: DA3 comes from its package (depth_anything_3), VGGT-Omega from its code (vggt_omega, importable or under
NOPE_GENERATORS/vggt_omega/code), the weights from NOPE_GENERATORS (see WEIGHTS). torch is imported only when a model runs.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import sys

import numpy as np

from nopeworld.paths import generators_path

BACKENDS = ("vggt_omega", "da3")
DEFAULT_BACKEND = "vggt_omega"
WEIGHTS = {"da3": ("NOPE_DA3_WEIGHTS", ("da3", "DA3-GIANT-1.1")),
           "vggt_omega": ("NOPE_VGGT_OMEGA_WEIGHTS", ("vggt_omega", "vggt_omega_1b_416_reproduce.pt"))}
VGGT_OMEGA_CODE = ("vggt_omega", "code")
VGGT_OMEGA_RES = 416
VGGT_OMEGA_PATCH = 16


def weights(backend):
    variable, parts = WEIGHTS[backend]
    return os.environ.get(variable) or str(generators_path(*parts))


@dataclass
class Reconstruction:
    frames: np.ndarray        # [N] video frame indices that were reconstructed
    depth: np.ndarray         # [N, h, w] metres
    conf: np.ndarray          # [N, h, w]
    c2w: np.ndarray           # [N, 4, 4] camera-to-world, world = hand-off camera, metres
    intrinsics: np.ndarray    # [N, 3, 3] at depth resolution
    scale: float              # metres per model unit
    history_fit: dict         # how well the history matches its true poses after scaling


def sample_frames(handoff, n_total, *, history_s=2.0, history_fps=4, gen_fps=8, fps=16):
    """The reconstructed frames: the last history_s of the history at history_fps, every frame at gen_fps after the hand-off."""
    hist = np.arange(handoff, max(0, handoff - int(history_s * fps)) - 1, -fps // history_fps)[::-1]
    gen = np.arange(handoff + fps // gen_fps, n_total, fps // gen_fps)
    return np.unique(np.r_[hist, gen]).astype(int)


_MODELS = {}


def da3(device="cuda"):
    if "da3" not in _MODELS:
        from depth_anything_3.api import DepthAnything3
        _MODELS["da3"] = DepthAnything3.from_pretrained(weights("da3")).to(device).eval()
    return _MODELS["da3"]


def omega_code():
    """Make the vggt_omega package importable when it is not installed."""
    try:
        import vggt_omega  # noqa: F401
    except ImportError:
        code = str(generators_path(*VGGT_OMEGA_CODE))
        if code not in sys.path:
            sys.path.append(code)


def vggt_omega(device="cuda", autocast=True):
    """VGGT-Omega 1B (416 checkpoint) in eval mode, loaded once; autocast: the model's own mixed precision."""
    if "vggt_omega" not in _MODELS:
        import torch
        omega_code()
        from vggt_omega.models import VGGTOmega
        model = VGGTOmega(autocast=autocast)
        model.load_state_dict(torch.load(weights("vggt_omega"), map_location="cpu", weights_only=True))
        _MODELS["vggt_omega"] = model.to(device).eval()
    return _MODELS["vggt_omega"]


def omega_input_shape(h, w, resolution=VGGT_OMEGA_RES, patch=VGGT_OMEGA_PATCH):
    """(height, width) of the 'balanced' preprocessing of an h x w picture: about resolution^2 pixels in whole patches at its aspect
    ratio. Pictures outside the aspect range [0.5, 2] would be centre-cropped there, cutting the field of view: refused."""
    aspect = h / max(w, 1)
    if not 0.5 <= aspect <= 2.0:
        raise ValueError(f"a {w} x {h} picture is outside VGGT-Omega's aspect range: it would be cropped")
    tokens = (resolution // patch) ** 2
    w_p = np.sqrt(tokens / aspect)
    h_p = tokens / w_p
    return max(1, int(np.round(h_p))) * patch, max(1, int(np.round(w_p))) * patch


def omega_inputs(frames_rgb, resolution=VGGT_OMEGA_RES):
    """[N, 3, H, W] float32 in [0, 1] from HxWx3 uint8 RGB pictures (PIL bicubic resize, then ToTensor); the model normalises itself."""
    import torch
    from PIL import Image
    h, w = frames_rgb[0].shape[:2]
    if any(f.shape[:2] != (h, w) for f in frames_rgb):
        raise ValueError("all frames of one reconstruction must have the same size")
    th, tw = omega_input_shape(h, w, resolution)
    out = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(f)).resize((tw, th), Image.Resampling.BICUBIC),
                               dtype=np.float32) / 255.0 for f in frames_rgb])
    return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()


def omega_predict(frames_rgb, device=None):
    """VGGT-Omega's raw output for all frames in one pass: (w2c [N, 3, 4], depth [N, H, W], depth confidence, intrinsics [N, 3, 3])."""
    import torch
    omega_code()
    from vggt_omega.utils.pose_enc import encoding_to_camera
    model = vggt_omega() if device is None else vggt_omega(device)
    x = omega_inputs(frames_rgb).to(next(model.parameters()).device)
    with torch.inference_mode():
        pred = model(x)
    ext, K = encoding_to_camera(pred["pose_enc"].float(), tuple(x.shape[-2:]))
    depth = pred["depth"][0].float()
    if depth.ndim == 4:                                     # [N, H, W, 1]
        depth = depth[..., 0]
    conf = pred.get("depth_conf")
    conf = None if conf is None else conf[0].float().reshape(depth.shape).cpu().numpy()
    return ext[0].cpu().numpy(), depth.cpu().numpy(), conf, K[0].cpu().numpy()


def metric_in_handoff_frame(w2c, depth, conf, intrinsics, frame_idx, gt_c2w_history, handoff):
    """A backend's raw multi-view output in the hand-off camera's frame, in metres by the true history's path length. w2c: [N, 3 or 4, 4]
    camera-from-world in the backend's units; gt_c2w_history None keeps the backend's units (scale 1, no history fit)."""
    w2c = np.asarray(w2c, float)
    if w2c.shape[1] == 3:
        w2c = np.concatenate([w2c, np.tile([[[0, 0, 0, 1.0]]], (len(w2c), 1, 1))], 1)
    c2w = np.linalg.inv(w2c)
    h = int(np.where(frame_idx == handoff)[0][0])
    rel = np.linalg.inv(c2w[h]) @ c2w                       # in the hand-off camera frame, the backend's units
    depth = np.asarray(depth, float)
    conf = None if conf is None else np.asarray(conf, float)
    intrinsics = np.asarray(intrinsics, float)
    if gt_c2w_history is None:
        return Reconstruction(frames=frame_idx, depth=depth, conf=conf, c2w=rel, intrinsics=intrinsics, scale=1.0, history_fit={})
    hist = np.where(frame_idx <= handoff)[0]
    gt_rel = np.linalg.inv(gt_c2w_history[handoff]) @ gt_c2w_history[frame_idx[hist]]
    est_len = np.linalg.norm(np.diff(rel[hist, :3, 3], axis=0), axis=1).sum()
    gt_len = np.linalg.norm(np.diff(gt_rel[:, :3, 3], axis=0), axis=1).sum()
    scale = gt_len / est_len if est_len > 1e-8 else float("nan")
    rel[:, :3, 3] *= scale
    err = np.linalg.norm(rel[hist, :3, 3] - gt_rel[:, :3, 3], axis=1)
    fit = dict(history_frames=int(len(hist)), true_path_m=round(float(gt_len), 3),
               position_error_max_m=round(float(err.max()), 3), position_error_mean_m=round(float(err.mean()), 3))
    return Reconstruction(frames=frame_idx, depth=depth * scale, conf=conf, c2w=rel, intrinsics=intrinsics, scale=float(scale), history_fit=fit)


def reconstruct(frames_rgb, frame_idx, gt_c2w_history, handoff, *, backend=DEFAULT_BACKEND, process_res=504):
    """frames_rgb: HxWx3 uint8 RGB pictures of frame_idx; gt_c2w_history: [handoff + 1, 4, 4] true poses (m), or None to keep the model's
    own units. backend: vggt_omega (the Task 2 evaluator) or da3 (the external depth baseline; process_res applies to DA3 only)."""
    frame_idx = np.asarray(frame_idx)
    if backend == "vggt_omega":
        w2c, depth, conf, K = omega_predict(list(frames_rgb))
        return metric_in_handoff_frame(w2c, depth, conf, K, frame_idx, gt_c2w_history, handoff)
    if backend != "da3":
        raise ValueError(f"reconstruction backend {backend!r} is not one of {BACKENDS}")
    import torch
    model = da3()
    with torch.no_grad():
        pred = model.inference(list(frames_rgb), process_res=process_res)
    return metric_in_handoff_frame(pred.extrinsics, pred.depth, pred.conf, pred.intrinsics, frame_idx, gt_c2w_history, handoff)


def backproject(depth, K, mask=None, stride=2):
    """Camera-frame 3-D points (m) of the pixels of a depth map, optionally inside a mask."""
    h, w = depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    d = depth[::stride, ::stride]
    keep = np.isfinite(d) & (d > 0)
    if mask is not None:
        keep &= mask[::stride, ::stride]
    x = (u[keep] - K[0, 2]) / K[0, 0] * d[keep]
    y = (v[keep] - K[1, 2]) / K[1, 1] * d[keep]
    return np.stack([x, y, d[keep]], 1)


def fit_floor(points, *, iters=400, rel_tol=0.02, max_tilt_deg=15.0, min_inliers=0.1, seed=0):
    """Ground plane of camera-frame points: RANSAC over planes within max_tilt_deg of horizontal (up is -y) below the camera, tolerance
    proportional to each point's distance. (n, d, inlier fraction) with n the unit normal pointing up and d > 0 the camera's height, or None."""
    if len(points) < 50:
        return None
    rng = np.random.default_rng(seed)
    dist = np.linalg.norm(points, axis=1)
    best = None
    for _ in range(iters):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n = n / np.linalg.norm(n)
        if n[1] > 0:
            n = -n                                          # point up (-y)
        if np.degrees(np.arccos(np.clip(-n[1], -1, 1))) > max_tilt_deg:
            continue
        d = -n @ a
        if d <= 0:                                          # the plane must be below the camera
            continue
        inl = np.abs(points @ n + d) < rel_tol * dist
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.mean() < min_inliers:
        return None
    q = points[best]
    centre = q.mean(0)
    n = np.linalg.svd(q - centre)[2][-1]
    if n[1] > 0:
        n = -n
    d = -n @ centre
    return (n, float(d), float(best.mean())) if d > 0 else None


def floor_points(depth, K, exclude=None, *, top=0.55, stride=2, max_points=20000, seed=0):
    """Camera-frame points of the lower part of a depth map (rows below `top` of the height), outside `exclude`."""
    h, w = depth.shape
    keep = np.zeros((h, w), bool)
    keep[int(top * h):] = True
    if exclude is not None:
        keep &= ~exclude
    pts = backproject(depth, K, keep, stride=stride)
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(seed).choice(len(pts), max_points, replace=False)]
    return pts


def level_and_scale(recon, origin, eye_m, obstacle=None, upto=None, agree_deg=8.0):
    """A reconstruction in its own units brought to metres by the eye height and turned level: the floor is fitted in each frame up to
    index `upto` (outside `obstacle` for frame `origin`); fits more than agree_deg from the median normal are dropped; the median of
    eye_m / height is the scale. (scale, R_level, info), or (None, None, info) when no frame shows the floor."""
    fits = []
    for i in range(len(recon.frames) if upto is None else upto + 1):
        pts = floor_points(recon.depth[i], recon.intrinsics[i], obstacle if i == origin else None)
        f = fit_floor(pts)
        if f is not None:
            n_i, d_i, frac = f
            fits.append((i, recon.c2w[i][:3, :3] @ n_i, d_i, frac))   # normal in the reference frame
    info = dict(floor_frames=len(fits), frames=int(len(recon.frames) if upto is None else upto + 1))
    if not fits:
        return None, None, info
    n_med = np.median([f[1] for f in fits], 0)
    n_med /= np.linalg.norm(n_med)
    kept = [f for f in fits if np.degrees(np.arccos(np.clip(f[1] @ n_med, -1, 1))) <= agree_deg]
    if not kept:
        return None, None, info
    scale = float(np.median([eye_m / d for _, _, d, _ in kept]))
    n_up = np.median([f[1] for f in kept], 0)
    n_up /= np.linalg.norm(n_up)
    target = np.array([0.0, -1.0, 0.0])
    v, c = np.cross(n_up, target), float(n_up @ target)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R_level = np.eye(3) + vx + vx @ vx / (1 + c) if c > -0.999 else np.diag([1.0, -1.0, -1.0])
    spread = [eye_m / d for _, _, d, _ in kept]
    info.update(kept=len(kept), tilt_deg=round(float(np.degrees(np.arccos(np.clip(c, -1, 1)))), 2),
                scale_spread=round(float(np.std(spread) / np.mean(spread)), 3),
                inlier_fraction=round(float(max(f[3] for f in kept)), 3))
    return scale, R_level, info


def apply_level_and_scale(recon, scale, R_level):
    """Copy of `recon` in metres with a horizontal floor: depth x scale; world turned by R_level (cameras keep their own frames)."""
    c2w = recon.c2w.copy()
    c2w[:, :3, 3] *= scale
    T = np.eye(4)
    T[:3, :3] = R_level
    return Reconstruction(frames=recon.frames, depth=recon.depth * scale, conf=recon.conf, c2w=T @ c2w,
                          intrinsics=recon.intrinsics, scale=float(scale), history_fit=recon.history_fit)
