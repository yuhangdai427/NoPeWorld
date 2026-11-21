"""The Task 2 evaluator on a run folder: each video on the 16 fps clock, reconstructed (VGGT-Omega) and tracked (SAM 2), then judged by
the NoPeBench rule (its open-route form on the open routes) against its route.

    <run>/<route_id>.mp4, <route_id>.meta.json          the generator's output (nopeworld-infer)
    <run>/task2_eval/video16/<route_id>.mp4             the video on the 16 fps clock (native_clock)
    <run>/task2_eval/measure/<route_id>.npz             the measurement (GPU part): camera path, depth, the obstacle's track
    <run>/task2_eval/verdicts.json                      {route_id: verdict} (CPU part)
(task2_eval_da3/ with the DA3 reconstruction.)

Only the generated camera path is estimated; the obstacle's points come from the route's true hand-off depth. When the model saw a crop
of the picture (`crop` in its meta), the real history frames and the mask are cut to that box first.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from . import native_clock
from .metrics import slide_of, wall_in_camera
from .reconstruction import DEFAULT_BACKEND, backproject, reconstruct, sample_frames
from .rule import band_points, outcome, outcome_control, signals, volume_points
from .scene import SceneField

GEN_FRAMES = 80             # generated frames per route (5 s at 16 fps)
K_NATIVE = np.array([[640.0, 0, 640.0], [0, 640.0, 360.0], [0, 0, 1]])   # Unreal Engine capture, FOV 90 at 1280 x 720
EVAL_DIR = "task2_eval"     # task2_eval_da3 for the DA3 reconstruction


def eval_dir_name(backend=DEFAULT_BACKEND):
    """The run's evaluation folder of one reconstruction backend (verdicts of two backends never share a folder)."""
    return EVAL_DIR if backend == DEFAULT_BACKEND else f"{EVAL_DIR}_{backend}"


def read_video(path):
    import cv2
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def package_context(pkg, case_json):
    """What the rule needs of a 16 fps route package; None (with a reason) if it is not scored."""
    import cv2
    pkg = Path(pkg)
    meta = json.loads((pkg / "meta.json").read_text())
    ev = json.loads((pkg / "eval" / "meta.json").read_text())
    if not meta.get("valid", True):
        return None, meta.get("invalid_reason")
    kh = meta["handoff_frame"]
    poses = np.load(pkg / "poses.npy").astype(float)
    rel = np.linalg.inv(poses[kh]) @ poses[min(kh + 16, len(poses) - 1)]
    u = rel[:3, 3].copy()
    u[1] = 0.0
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([0.0, 0.0, 1.0])
    ref = np.linalg.inv(poses[kh]) @ np.load(pkg / "eval" / "reference_poses.npy").astype(float)
    common = dict(kh=kh, hist=poses[:kh + 1], eye=ev["eye_m"], u=u, speed=ev["speed_mps"], case_json=Path(case_json),
                  command=np.linalg.inv(poses[kh]) @ poses[kh:kh + GEN_FRAMES + 1], reference=ref[:, :3, 3])
    if ev["kind"] == "control":
        return dict(common, control=True, mask=None, depth_true=None), None
    if ev["kind"] != "critical":
        return None, f"route kind {ev['kind']} is not scored"
    n_true, off_true = wall_in_camera(ev["wall_point_xy"], ev["wall_normal_xy"], poses[kh])
    mask = cv2.imread(str(pkg / "eval" / "handoff_mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    return dict(common, control=False, mask=mask, n_true=n_true, beta=ev["approach_angle_deg"], feasible=slide_of(ref, n_true, off_true, u),
                depth_true=pkg / "eval" / "handoff_depth.npy"), None


def route_context(route_dir):
    """package_context of a route folder of the dataset (fps16/ and case/case.json)."""
    route_dir = Path(route_dir)
    return package_context(route_dir / "fps16", route_dir / "case" / "case.json")


def whole_box(crop, where=""):
    """A box [left, top, right, bottom] in ints; one not in whole pixels raises."""
    if not (len(crop) == 4 and all(float(x).is_integer() for x in crop)):
        raise ValueError(f"{where}crop {crop} is not a box of whole pixels")
    return [int(x) for x in crop]


def video_crop(video):
    """The part of the native picture that reached the model, from <route>.meta.json next to <route>.mp4; None without one."""
    video = Path(video)
    path = video.with_name(video.stem + ".meta.json")
    if not path.exists():
        return None
    crop = json.loads(path.read_text()).get("crop")
    return None if crop is None else whole_box(crop, f"{path}: ")


def crop_picture(picture, crop):
    """A native picture (frame or mask) cut to the box crop; with crop None the picture itself."""
    if crop is None:
        return picture
    left, top, right, bottom = whole_box(crop)
    h, w = picture.shape[:2]
    if not (0 <= left < right <= w and 0 <= top < bottom <= h):
        raise ValueError(f"crop {[left, top, right, bottom]} does not lie inside a picture of {w} x {h}")
    return picture[top:bottom, left:right]


def prepare_frame(frame, size, crop=None):
    """A real frame (BGR) as the generated video shows it: the part the model saw, at size (width, height)."""
    import cv2
    return cv2.resize(crop_picture(frame, crop), size, interpolation=cv2.INTER_AREA)


def prepare_mask(mask, size, crop=None):
    """The native hand-off mask as the generated video shows it."""
    import cv2
    mask = crop_picture(mask, crop)
    if mask.shape != (size[1], size[0]):
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


def load_frames(video, kh, history=None):
    """All frames of a test video at its own resolution; a video of the generated frames only (at most 81) gets the real history[:kh + 1],
    cut to the video's crop, in front."""
    import cv2
    crop = video_crop(video)
    frames = read_video(video)
    if history is not None and len(frames) <= GEN_FRAMES + 1:
        frames = frames[len(frames) - GEN_FRAMES:]
        size = (frames[0].shape[1], frames[0].shape[0])
        frames = [prepare_frame(f, size, crop) for f in history[:kh + 1]] + frames
    size = (frames[-1].shape[1], frames[-1].shape[0])
    return [f if (f.shape[1], f.shape[0]) == size else cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]


def image_structure(frames):
    """[n, 3] per frame: mean gradient magnitude, grey standard deviation, share of flat 20 x 20 blocks (at 320 x 180)."""
    import cv2
    out = []
    for f in frames:
        g = cv2.cvtColor(cv2.resize(f, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
        flat = g.reshape(9, 20, 16, 20).std(axis=(1, 3)) < 2 / 255
        out.append([float(mag.mean()), float(g.std()), float(flat.mean())])
    return np.array(out, np.float32)


def measure(video, ctx, history, backend=DEFAULT_BACKEND):
    """The GPU part of scoring one video: the reconstruction (metric through the true history) and the obstacle's track from the hand-off mask."""
    import cv2
    from .track import track
    kh, crop = ctx["kh"], video_crop(video)
    frames = load_frames(video, kh, history)
    size = (frames[-1].shape[1], frames[-1].shape[0])
    mask = None if ctx["mask"] is None else prepare_mask(ctx["mask"], size, crop)
    idx = sample_frames(kh, len(frames))
    t0 = time.time()
    recon = reconstruct([cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in idx], idx, ctx["hist"], kh, backend=backend)
    t1 = time.time()
    if mask is None:            # an open route: no obstacle to track
        masks, scores = [np.zeros((size[1], size[0]), bool)] * (len(frames) - kh), np.full(len(frames) - kh, -1e3)
    else:
        masks, scores = track(frames, kh, mask)
    t2 = time.time()
    small = np.array([cv2.resize(m.astype(np.uint8), (160, 90), interpolation=cv2.INTER_NEAREST) for m in masks], bool)
    m = dict(frames=np.asarray(recon.frames), c2w=recon.c2w, intrinsics=recon.intrinsics, depth=recon.depth.astype(np.float16),
             scale=float(recon.scale), history_fit=json.dumps(recon.history_fit), scores=np.asarray(scores, float),
             area=np.array([x.mean() for x in masks]), border=np.array([bool(x[0].any() or x[-1].any() or x[:, 0].any() or x[:, -1].any()) for x in masks]),
             masks_small=np.packbits(small, axis=-1), n_frames=len(frames), size=np.array(size), structure=image_structure(frames[kh:]),
             seconds=np.array([t1 - t0, t2 - t1]), recon=np.array(backend))
    if recon.conf is not None:
        m["conf"] = np.clip(np.asarray(recon.conf), -65504.0, 65504.0).astype(np.float16)
    if crop is not None:
        m["crop"] = np.array(crop)
    return m


def save_measure(path, m):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **m)
    tmp.replace(path)


def load_measure(path):
    z = np.load(path)
    m = {k: z[k] for k in z.files}
    m["scale"] = float(m["scale"])
    m["history_fit"] = json.loads(str(m["history_fit"]))
    m["n_frames"] = int(m["n_frames"])
    m["recon"] = str(m["recon"]) if "recon" in m else "da3"
    return m


def judge(m, ctx, scenes_root):
    """The rule on one video's measurement (load_measure) against its route's context."""
    kh = ctx["kh"]
    idx = np.asarray(m["frames"])
    recon = SimpleNamespace(frames=idx, c2w=np.asarray(m["c2w"], float), depth=m["depth"], intrinsics=np.asarray(m["intrinsics"], float))
    scene = SceneField(ctx["case_json"], ctx["hist"][kh], scenes_root)
    noise = (m["history_fit"] or {}).get("position_error_max_m")
    if ctx.get("control"):
        sig = signals(recon, np.zeros((0, 3)), ctx["u"], m["scores"], kh, area=m["area"], border=m["border"], scene=scene,
                      command=ctx["command"], reference=ctx["reference"])
        res = outcome_control(sig, ctx["u"], ctx["speed"], noise_m=noise)
        res.update(handoff_free_space_m2=round(scene.handoff_free_m2, 1))
    else:
        if ctx["depth_true"] is None or not Path(ctx["depth_true"]).exists():
            return dict(error="the rule needs the true hand-off depth")
        hand = np.load(ctx["depth_true"]).astype(float)
        pts = volume_points(backproject(hand, K_NATIVE, ctx["mask"]))
        if len(pts) < 20:
            return dict(error="no obstacle points at the hand-off frame", points=int(len(pts)))
        sig = signals(recon, pts, ctx["u"], m["scores"], kh, area=m["area"], border=m["border"], scene=scene, command=ctx["command"],
                      reference=ctx["reference"], pts_all=band_points(backproject(hand, K_NATIVE)))
        res = outcome(sig, ctx["n_true"], ctx["u"], ctx["beta"], ctx["speed"], ctx["feasible"], noise_m=noise)
        res.update(handoff_free_space_m2=round(scene.handoff_free_m2, 1), obstacle_points=int(len(pts)))
    sec = np.asarray(m["seconds"], float)
    res.update(scale=round(float(m["scale"]), 4), history_fit=m["history_fit"], recon=m["recon"],
               seconds={m["recon"]: round(float(sec[0]), 1), "tracker": round(float(sec[1]), 1)})
    if "crop" in m:
        res["crop"] = [int(x) for x in m["crop"]]
    return res


def evaluate_run(run, route_dirs, scenes_root, *, stages=("convert", "measure", "judge"), backend=DEFAULT_BACKEND, log=print):
    """The evaluator on a run folder (route_dirs: {route_id: route folder}); of the stages only measure needs the GPU. Videos already
    converted, measured or judged are kept. Returns the verdicts."""
    run = Path(run)
    out = run / eval_dir_name(backend)
    v16, meas = out / "video16", out / "measure"
    verdict_file = out / "verdicts.json"
    verdicts = json.loads(verdict_file.read_text()) if verdict_file.is_file() else {}
    for route_id, route_dir in sorted(route_dirs.items()):
        video = run / f"{route_id}.mp4"
        if not video.is_file():
            continue
        try:
            if "convert" in stages and not (v16 / video.name).is_file():
                native_clock.convert(video, v16)
            npz = meas / f"{route_id}.npz"
            if "measure" in stages and not npz.is_file():
                ctx, why = route_context(route_dir)
                if ctx is None:
                    verdicts[route_id] = dict(invalid=why)
                    continue
                save_measure(npz, measure(v16 / video.name, ctx, read_video(Path(route_dir) / "fps16" / "history.mp4"), backend=backend))
            if "judge" in stages and npz.is_file() and "category" not in verdicts.get(route_id, {}):
                ctx, why = route_context(route_dir)
                m = load_measure(npz)
                if m["recon"] != backend:
                    raise ValueError(f"{npz} is a {m['recon']} measurement, this run judges {backend}")
                verdicts[route_id] = judge(m, ctx, scenes_root) if ctx is not None else dict(invalid=why)
                log(route_id, json.dumps({k: verdicts[route_id].get(k) for k in ("category", "violation", "progress")}))
        except Exception as error:              # a video that cannot be scored says so; the others go on
            verdicts[route_id] = dict(error=repr(error)[:400])
            log(route_id, "error", repr(error)[:200])
        out.mkdir(parents=True, exist_ok=True)
        tmp = verdict_file.with_name(verdict_file.name + ".tmp")
        tmp.write_text(json.dumps(dict(sorted(verdicts.items())), indent=1))
        tmp.replace(verdict_file)
    return verdicts
