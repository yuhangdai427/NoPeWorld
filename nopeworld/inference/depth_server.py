"""The DA3 process of the external depth baseline, in the environment that has DA3 (depth_anything_3) and OpenCV; loaded once, asked over
pipes by nopeworld.inference.methods.external_depth, one request at a time while the generation waits; end of input ends it.

    python -m nopeworld.inference.depth_server [--offload]

A request is one line of JSON, then its frames as raw bytes (uint8 RGB [T, H, W, 3]):
    {"op": "sweep", "mode": fpv | tpv | wbench, "case", "package", "size": [w, h], "crop", "offsets", "disps_cam", "tail", "bytes", "shape"}
The frames are the decoded frames after the hand-off at `offsets` (increasing; none before the first generated chunk), the latest last.
fpv / tpv: the route package's real history (prepared at the generated size) and its true path give the metres and the vertical; wbench:
one start image, no history, metres and level from the floor at an eye height of 1.6 m. disps_cam [C, 3]: each candidate's displacement
over the next chunk in the last frame's camera, KEEP first. The answer is one line of JSON: h, P, A, in_view, slide (depth_sweep), DA3's
scale and its fit on the history, timings; a history that does not move gives no metres (no_scale, h / P / A / in_view None); a failed
request is answered with error and trace.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback
from unittest import mock

import numpy as np

from .depth_sweep import TAIL, body_scores, level_rotation, scores

EYE_M = 1.6           # wbench (image-to-video cases): the eye height the floor fit assumes


class Cases:
    """A route's real frames and known path, kept while requests are about the same package."""

    def __init__(self):
        self.name, self.value = None, None

    def of(self, request):
        if self.name == request["package"]:
            return self.value
        import cv2
        from nopebench.task2.evaluate import prepare_frame, read_video
        from nopebench.task2.reconstruction import sample_frames
        package = Path(request["package"])
        kh = json.loads((package / "meta.json").read_text())["handoff_frame"]
        poses = np.load(package / "poses.npy").astype(float)[:kh + 1]
        indices = sample_frames(kh, kh + 1)
        video = read_video(package / "history.mp4")
        size = tuple(request["size"])          # (width, height)
        real = [cv2.cvtColor(prepare_frame(video[i], size, request["crop"]), cv2.COLOR_BGR2RGB) for i in indices]
        self.name = request["package"]
        self.value = dict(kh=kh, poses=poses, indices=indices, real=real)
        return self.value


def serve(requests, replies, handle):
    """The loop: requests and replies binary file objects; handle(request, frames) -> a dict. An exception in handle is answered."""
    while True:
        line = requests.readline()
        if not line:
            return
        request = json.loads(line)
        if request.get("op") == "quit":
            return
        size = int(request.get("bytes", 0))
        raw = requests.read(size) if size else b""
        if len(raw) != size:
            return
        frames = np.frombuffer(raw, np.uint8).reshape(request["shape"]).copy() if size else None
        try:
            reply = handle(request, frames)
        except Exception as exc:
            reply = dict(error=repr(exc), trace=traceback.format_exc()[-1500:])
        replies.write((json.dumps(reply) + "\n").encode())
        replies.flush()


class FloorPool:
    """reconstruction.level_and_scale with its per-frame floor fits run side by side in spawned workers; the result is the same, each fit
    answered only for the very points it was fitted on."""

    def __init__(self, workers=12):
        self.workers, self.pool = workers, None

    def level_and_scale(self, recon, origin, eye_m, obstacle=None, upto=None, agree_deg=8.0):
        from nopebench.task2 import reconstruction as G
        n = len(recon.frames) if upto is None else upto + 1
        points = [G.floor_points(recon.depth[i], recon.intrinsics[i], obstacle if i == origin else None) for i in range(n)]
        if self.pool is None:
            self.pool = mp.get_context("spawn").Pool(self.workers)
        fits = self.pool.map(G.fit_floor, points, chunksize=1)
        queue = list(zip(points, fits))

        def fitted(pts, **kwargs):
            if kwargs or not queue:
                raise RuntimeError("level_and_scale asked for a floor fit this pool did not make")
            want, fit = queue.pop(0)
            if pts.shape != want.shape or not np.array_equal(pts, want):
                raise RuntimeError("level_and_scale asked for the floor of other points than the ones fitted")
            return fit

        with mock.patch.object(G, "fit_floor", fitted):
            out = G.level_and_scale(recon, origin, eye_m, obstacle=obstacle, upto=upto, agree_deg=agree_deg)
        if queue:
            raise RuntimeError(f"{len(queue)} floor fits not asked for")
        return out

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None


def level_of(poses_kh):
    """R_level from the true hand-off camera (c2w, world z up): its frame turned so that the world's up is OpenCV's (0, -1, 0)."""
    return level_rotation(np.asarray(poses_kh, float)[:3, :3].T @ np.array([0.0, 0.0, 1.0]))


def body_of(package, poses, kh, R_level):
    """The third-person body at the hand-off: its root in the camera's frame, its heading against the camera's on the level ground plane
    (rad), the entity, its box and root_up (the package's eval/meta.json and eval/body_poses.npy)."""
    ev = json.loads((Path(package) / "eval" / "meta.json").read_text())
    body = np.load(Path(package) / "eval" / "body_poses.npy").astype(float)[kh]
    R, t = np.asarray(poses[kh], float)[:3, :3], np.asarray(poses[kh], float)[:3, 3]
    o = R.T @ (body[:3] * np.array([1.0, -1.0, 1.0]) - t)
    yaw = np.radians(body[3])
    fwd_body = R_level @ (R.T @ np.array([np.cos(yaw), -np.sin(yaw), 0.0]))      # Unreal yaw -> the poses' world -> the level camera
    fwd_cam = R_level @ np.array([0.0, 0.0, 1.0])
    a, b = fwd_cam[[0, 2]], fwd_body[[0, 2]]
    turn = float(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b))
    return dict(o_cam=o, body_turn=turn, entity=ev["entity"], box={k: float(v) for k, v in ev["body_box"].items()}, root_up=float(ev.get("root_up", 0.0)))


def wbench_case(request):
    """An image-to-video package (no history): its one image at the generated size; no true pose."""
    import cv2
    from nopebench.task2.evaluate import prepare_frame
    picture = cv2.imread(str(Path(request["package"]) / "handoff.png"))
    return dict(kh=0, indices=np.array([0]), real=[cv2.cvtColor(prepare_frame(picture, tuple(request["size"]), request.get("crop")), cv2.COLOR_BGR2RGB)], poses=None)


def answer(request, frames, cases, reconstruct, level_and_scale=None):
    """One request. reconstruct(pictures, numbers, poses, kh) and level_and_scale(recon, origin, eye_m): the DA3 reconstruction and the
    floor fit (injected, so the rule can be checked without DA3)."""
    mode = request.get("mode", "fpv")
    if mode not in ("fpv", "tpv", "wbench"):
        raise ValueError(f"mode {mode!r}: fpv, tpv or wbench")
    case = wbench_case(request) if mode == "wbench" else cases.of(request)
    kh = case["kh"]
    offsets = np.asarray(request.get("offsets") or [], int)
    n = 0 if frames is None else len(frames)
    if n != len(offsets) or (len(offsets) and (offsets.min() < 1 or np.any(np.diff(offsets) <= 0))):
        raise ValueError(f"{n} frames for offsets {offsets.tolist()}: offsets after the hand-off, increasing, one per frame")
    numbers = np.r_[case["indices"], kh + offsets].astype(int)
    pictures = case["real"] + ([f for f in frames] if n else [])
    began = time.perf_counter()
    recon = reconstruct(pictures, numbers, case["poses"], kh)
    seconds = time.perf_counter() - began
    base = dict(seconds_da3=round(seconds, 2), reconstructed_frames=int(len(numbers)), generated_frames=int(n), mode=mode)
    none = dict(base, no_scale=True, scale=None, h=None, P=None, A=None, in_view=None)
    tail = min(int(request.get("tail", TAIL)), len(numbers))
    disps = np.asarray(request["disps_cam"], float)
    if mode == "wbench":
        from nopebench.task2.reconstruction import apply_level_and_scale
        scale, R_level, info = level_and_scale(recon, 0, EYE_M)
        if scale is None:
            return dict(none, floor=info)
        recon = apply_level_and_scale(recon, scale, R_level)
        got, how = scores(np.asarray(recon.depth, np.float32), np.asarray(recon.intrinsics, float), np.asarray(recon.c2w, float), disps, tail=tail)
        how.update(floor=info)
    else:
        if not np.isfinite(recon.scale):
            return none
        R_level = level_of(case["poses"][kh])
        args = (np.asarray(recon.depth, np.float32), np.asarray(recon.intrinsics, float), np.asarray(recon.c2w, float), disps)
        if mode == "tpv":
            body = body_of(request["package"], case["poses"], kh, R_level)
            got, how = body_scores(*args, R_level=R_level, tail=tail, **body)
            how.update(o_cam=[round(float(x), 3) for x in body["o_cam"]], body_turn_deg=round(float(np.degrees(body["body_turn"])), 2), entity=body["entity"])
        else:
            got, how = scores(*args, R_level=R_level, tail=tail)
    return dict(base, **how, h=[round(float(x), 4) for x in got["h"]], P=[round(float(x), 6) for x in got["P"]], A=[round(float(x), 6) for x in got["A"]],
                in_view=[bool(x) for x in got["in_view"]], slide=[round(float(x), 4) for x in got["slide"]], no_scale=False,
                scale=round(float(recon.scale), 4), history_error_max_m=recon.history_fit.get("position_error_max_m"),
                seconds_rule=round(time.perf_counter() - began - seconds, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offload", action="store_true", help="the weights wait in host memory between requests (more GPU room for the generation's peak)")
    args = ap.parse_args()
    replies = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)                               # what DA3 prints goes to the log, not into the answers
    import torch
    from nopebench.task2.reconstruction import da3, reconstruct
    began = time.perf_counter()
    model = da3()
    if args.offload:
        model.to("cpu")
    torch.cuda.empty_cache()
    cases = Cases()
    floors = FloorPool(int(os.environ.get("NOPE_FLOOR_WORKERS", "12")))

    def da3_reconstruct(pictures, numbers, poses, kh):
        return reconstruct(pictures, numbers, poses, kh, backend="da3")

    def handle(request, frames):
        if request.get("op") != "sweep":
            raise ValueError(f"op {request.get('op')!r}: this process answers 'sweep'")
        if args.offload:
            model.to("cuda")
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        result = answer(request, frames, cases, da3_reconstruct, floors.level_and_scale)
        result.update(peak_gib_da3=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), seconds_server=round(time.perf_counter() - started, 2))
        if args.offload:
            model.to("cpu")
        torch.cuda.empty_cache()
        return result
    replies.write((json.dumps(dict(ready=True, load_seconds=round(time.perf_counter() - began, 1), offload=args.offload,
                                   gib_resident=round(torch.cuda.memory_allocated() / 2 ** 30, 2))) + "\n").encode())
    replies.flush()
    try:
        serve(sys.stdin.buffer, replies, handle)
    finally:
        floors.close()


if __name__ == "__main__":
    main()
