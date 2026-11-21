"""Block taps, background preparation of the next videos, a background head scorer, and video I/O."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import threading

import numpy as np
import torch


def no_huge_pages():
    """NumPy's transparent-huge-page advice makes first touches of large arrays stall on fragmented memory; the bytes are the same."""
    try:
        from numpy._core import multiarray
    except ImportError:
        from numpy.core import multiarray
    multiarray._set_madvise_hugepage(False)


no_huge_pages()


def read_video(path, limit=None):
    """All frames of a video as uint8 [T, H, W, 3] RGB."""
    import imageio_ffmpeg
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    meta = next(reader)
    w, h = meta["size"]
    frames = []
    try:
        for data in reader:
            frames.append(np.frombuffer(data, np.uint8).reshape(h, w, 3))
            if limit is not None and len(frames) >= limit:
                break
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"could not decode video: {path}")
    return np.stack(frames)


def write_video(path, video, fps):
    """[3, T, H, W] in [-1, 1] to H.264 (quality 8), written under a temporary name first."""
    import imageio_ffmpeg
    path = Path(path)
    frames = ((video.detach().float().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 3, 0).cpu().numpy()
    tmp = path.parent / "_tmp" / path.name
    tmp.parent.mkdir(exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(tmp), (frames.shape[2], frames.shape[1]), fps=fps, codec="libx264", quality=8,
                                         macro_block_size=1, output_params=["-movflags", "+faststart"])
    writer.send(None)
    for frame in frames:
        writer.send(np.ascontiguousarray(frame))
    writer.close()
    tmp.replace(path)


class TokenTap:
    """Captures the video tokens of one forward at the given blocks, stacked [L, F, H, W, D].

    extract(output) returns a block's video tokens of the current chunk as [1, tokens, D]."""

    def __init__(self, blocks, layers, extract=None):
        self.layers = tuple(layers)
        self.extract = extract or (lambda output: output)
        self.active = False
        self.values = {}
        self.grid = None
        self.handles = [blocks[layer].register_forward_hook(self._hook(layer)) for layer in self.layers]

    def _hook(self, layer):
        def record(module, inputs, output):
            if not self.active:
                return
            tokens = self.extract(output)
            if not isinstance(tokens, torch.Tensor) or tokens.shape[0] != 1:
                raise ValueError("the tap supports single-stream, unsharded DiT output")
            frames, height, width = self.grid
            count = frames * height * width
            if tokens.shape[1] != count:
                raise ValueError(f"block {layer} has {tokens.shape[1]} video tokens, the grid {self.grid} needs {count}")
            self.values[layer] = tokens[0].detach().reshape(frames, height, width, tokens.shape[-1]).clone()
        return record

    def begin(self, grid):
        if self.active:
            raise RuntimeError("capture window already open")
        self.grid = tuple(grid)
        self.values = {}
        self.active = True

    def finish(self):
        self.active = False
        if set(self.values) != set(self.layers):
            raise RuntimeError(f"missing hidden layers: {sorted(set(self.layers) - set(self.values))}")
        hidden = torch.stack([self.values[layer] for layer in self.layers])
        self.values = {}
        if not torch.isfinite(hidden).all() or hidden.float().std() <= 1e-8:
            raise FloatingPointError("invalid DiT hidden state")
        return hidden

    def cancel(self):
        self.active = False
        self.values = {}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.cancel()


class Prefetcher:
    """fn(key) in worker threads for announced keys, at most `depth` results held; get(key) computes an unannounced key inline."""

    def __init__(self, fn, *, workers=1, depth=2):
        self.fn, self.depth = fn, depth
        self.pool = ThreadPoolExecutor(workers, thread_name_prefix="prefetch") if workers else None
        self.jobs: OrderedDict[str, Future] = OrderedDict()
        self.lock = threading.Lock()

    def announce(self, keys):
        if self.pool is None:
            return
        with self.lock:
            for key in keys[:self.depth]:
                if key not in self.jobs and len(self.jobs) <= self.depth:
                    self.jobs[key] = self.pool.submit(self.fn, key)

    def get(self, key):
        with self.lock:
            job = self.jobs.pop(key, None)
        return self.fn(key) if job is None else job.result()

    def discard(self, key):
        with self.lock:
            job = self.jobs.pop(key, None)
        if job is not None:
            job.cancel()

    def cancel_all(self):
        with self.lock:
            jobs, self.jobs = list(self.jobs.values()), OrderedDict()
        for job in jobs:
            job.cancel()

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None
        self.jobs.clear()


class AsyncScorer:
    """A head forward on a worker thread, so the generation goes on while it scores; the result equals the inline one."""

    def __init__(self, score):
        self.score = score
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="scorer")

    def submit(self, *args, **kwargs) -> Future:
        return self.pool.submit(self._run, args, kwargs)

    def _run(self, args, kwargs):
        with torch.inference_mode():
            return self.score(*args, **kwargs)

    def close(self):
        self.pool.shutdown(wait=True)
        self.score = None
