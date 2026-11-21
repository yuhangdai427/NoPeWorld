"""External depth baseline, generation side: before each decided chunk a DA3 process scores the 34 candidates' clearances on the
decoded frames so far, and the released collision head's decision rule keeps the command or stops (only KEEP and STOP can be chosen).

The DA3 process (`python -m nopeworld.inference.depth_server`, in the environment that has DA3, NOPE_DEPTH_PYTHON) reads the 16 fps route
packages: the real history and its true path give the scale. A 24 fps generator sends the frames that fall on the 16 fps clock exactly
(16 fps frame 2 i is 24 fps frame 3 i). Standard library and numpy only at import, so every generator environment can import it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
import sys

import numpy as np

from ...paths import REPO

EVAL_FPS = 16
TAIL = 2                                  # decoded frames whose depth is lifted
COMMITTED_S, CHUNK_S = 0.25, 0.125        # frames before the chunk every 0.25 s, the chunk's own every 0.125 s


def clock16(n_decoded, native_fps, *, first_is_handoff):
    """(end16, index_of): the last 16 fps frame after the hand-off that has a decoded frame at its instant, and its index in the decoded frames."""
    after = n_decoded - 1 if first_is_handoff else n_decoded
    if native_fps == EVAL_FPS:
        end16 = after

        def index_of(o16):
            return o16 if first_is_handoff else o16 - 1
    elif native_fps == 24:
        end16 = 2 * (after // 3)

        def index_of(o16):
            if o16 % 2:
                raise ValueError(f"16 fps frame {o16} has no 24 fps frame at its instant")
            o24 = 3 * (o16 // 2)
            return o24 if first_is_handoff else o24 - 1
    else:
        raise ValueError(f"native fps {native_fps}")
    return max(end16, 0), index_of


def chunk_len16(native_frames, native_fps):
    """A chunk of native_frames frames on the 16 fps clock, rounded down to even."""
    n = int(native_frames * EVAL_FPS / native_fps)
    return n - n % 2 if native_fps != EVAL_FPS else n


def to_rgb(video):
    """[3, T, H, W] in [-1, 1] (torch) -> uint8 [T, H, W, 3]."""
    return ((video.detach().float().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 3, 0).cpu().numpy()


def disps_in_camera(camera_c2w, ends_c2w, start_c2w=None):
    """[C, 3]: each candidate's displacement over the chunk, from start_c2w (default the camera) to its end pose, in the camera frame."""
    camera = np.asarray(camera_c2w, np.float64)
    start = camera if start_c2w is None else np.asarray(start_c2w, np.float64)
    d = np.asarray(ends_c2w, np.float64)[:, :3, 3] - start[:3, 3]
    return d @ camera[:3, :3]


def geo_choose(reply, selection, device="cpu"):
    """The decision rule on DA3's P (as logits, temperature 1) and A, with only KEEP and STOP choosable. None without scale."""
    import torch
    from ...head.candidates import NAMES
    from ...head.choose import choose
    if reply.get("P") is None:
        return None
    p = torch.as_tensor(np.asarray(reply["P"], float), dtype=torch.float32, device=device).clamp(1e-6, 1 - 1e-6)
    a = torch.as_tensor(np.asarray(reply["A"], float), dtype=torch.float32, device=device)[None]
    mask = torch.zeros(1, len(p), dtype=torch.bool, device=device)
    mask[0, [NAMES.index("KEEP"), NAMES.index("STOP")]] = True
    reply["stop_only"], reply["p_da3_all"] = True, [round(float(x), 4) for x in reply["P"]]
    sel = choose(torch.log(p / (1 - p))[None], a, mask, temperature=1.0, keep_margin=selection.get("keep_margin", 0.0),
                 rule=selection.get("rule", "product"), tau=selection.get("tau", 0.5))
    return {"choice": int(sel["choice"][0]), "p": sel["P"][0].float().cpu(), "r": sel["R"][0].float().cpu(), "a": a[0].float().cpu(),
            "clearance": torch.as_tensor(np.asarray(reply["h"], float), dtype=torch.float32)}


def whole_box(crop):
    """A crop box in whole pixels, as floats."""
    return None if crop is None else [float(round(float(x))) for x in crop]


class Da3Stop:
    """The DA3 process as a child; it ends with this process's pipes."""

    def __init__(self, log_path, *, offload=False, timeout=300.0, command=None):
        self.timeout = timeout
        python = os.environ.get("NOPE_DEPTH_PYTHON", sys.executable)
        env = dict(os.environ, PYTHONPATH=str(REPO), OMP_NUM_THREADS="4", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONFAULTHANDLER="1")
        self.log = open(log_path, "ab")
        self.process = subprocess.Popen(command or [python, "-W", "ignore", "-m", "nopeworld.inference.depth_server", *(["--offload"] if offload else [])],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=env, cwd=str(REPO))
        self.ready = self.read(timeout=max(timeout, 600.0))
        if not self.ready.get("ready"):
            raise RuntimeError(f"the DA3 process did not start: {self.ready}")

    def read(self, timeout=None):
        ready, _, _ = select.select([self.process.stdout], [], [], self.timeout if timeout is None else timeout)
        if not ready:
            self.process.kill()
            self.process.wait()
            raise TimeoutError(f"the DA3 process did not answer in {self.timeout if timeout is None else timeout} s; it is stopped")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"the DA3 process ended with code {self.process.poll()}")
        return json.loads(line)

    def ask(self, request, frames):
        frames = np.ascontiguousarray(frames, np.uint8)
        raw = frames.tobytes()
        self.process.stdin.write((json.dumps({**request, "bytes": len(raw), "shape": list(frames.shape)}) + "\n").encode())
        self.process.stdin.write(raw)
        self.process.stdin.flush()
        reply = self.read()
        if "error" in reply:
            raise RuntimeError(f'the DA3 process failed: {reply["error"]}\n{reply.get("trace", "")}')
        return reply

    def close(self):
        try:
            self.process.stdin.close()
            self.process.wait(timeout=30)
        except Exception:
            self.process.kill()
        self.log.close()


def offsets_to_send(chunk_end, chunk_len, fps):
    """The generated frames sent after the chunk ending at frame `chunk_end` (1-based after the hand-off) of `chunk_len` frames."""
    first = chunk_end - chunk_len
    step_c, step_k = max(int(round(COMMITTED_S * fps)), 1), max(int(round(CHUNK_S * fps)), 1)
    before = list(range(step_c, first + 1, step_c))
    inside = list(range(chunk_end, first, -step_k))[::-1]
    return sorted(set(before + inside))


def lingbot_offsets(chunk, history_chunks):
    """LingBot: the clip frames of chunk `chunk` (chunk 0 = frames 0-8, chunk k = 12k-3 .. 12k+8) as (first, last) frames after the hand-off."""
    first, last = (0, 8) if chunk == 0 else (12 * chunk - 3, 12 * chunk + 8)
    handoff = 12 * history_chunks - 4 if history_chunks else 0
    return first - handoff, last - handoff


class GeoSafety:
    """One route's external depth on the generation side; every answer is logged."""

    def __init__(self, da3, package, *, fps, crop=None, tail=TAIL, mode=None, case=None):
        """mode: fpv (with history), tpv, or wbench (image to video, metres from the floor); None reads the package's view."""
        self.da3, self.package, self.fps, self.crop, self.tail = da3, Path(package), float(fps), crop, tail
        self.case = case or self.package.name
        if mode is None:
            meta = self.package / "eval" / "meta.json"
            mode = "tpv" if meta.exists() and json.loads(meta.read_text()).get("view") == "TP" else "fpv"
        self.mode, self.size, self.asked = mode, None, []

    def scores(self, chunk_after, decoded_end, chunk_len, disps_cam, frames_by_offset):
        """Before generated chunk `chunk_after`: h, P, A, in_view of the candidates (None without scale)."""
        if self.size is None:
            raise ValueError("the generated frames' size is not known yet")
        offsets = offsets_to_send(decoded_end, chunk_len, self.fps) if decoded_end >= 1 else []
        frames = frames_by_offset(offsets) if offsets else np.zeros((0, self.size[1], self.size[0], 3), np.uint8)
        request = dict(op="sweep", mode=self.mode, case=self.case, package=str(self.package), size=list(self.size), crop=self.crop, offsets=offsets,
                       disps_cam=np.asarray(disps_cam, float).round(5).tolist(), tail=self.tail)
        reply = self.da3.ask(request, frames)
        self.asked.append(dict(chunk_after=int(chunk_after), frames_sent=len(offsets), **{k: v for k, v in reply.items() if k not in ("h", "P", "A", "in_view")}))
        return reply


class DepthSpec:
    """What the rollout hands an adapter for the external depth method: the DA3 process and, per route id, the 16 fps package
    it reads (a Task 2 route's fps16/ folder; a Task 3 case's own package)."""

    def __init__(self, da3, package_of):
        self.da3 = da3
        self.package_of = package_of if callable(package_of) else (lambda case, root=Path(package_of): root / case)

    def geo(self, case, crop, *, i2v):
        return GeoSafety(self.da3, self.package_of(case), fps=EVAL_FPS, crop=whole_box(crop), mode="wbench" if i2v else "fpv", case=case)
