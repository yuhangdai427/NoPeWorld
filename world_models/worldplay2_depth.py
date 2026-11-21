"""WorldPlay2's external depth method: before each generated chunk with a commanded move, DA3 scores the candidates on the frames decoded so far
and the decision rule keeps the command or stops; the change is written into the chunk's rows before any forward reads them.

The frames are the pipeline's own streaming decode of every finished chunk (WorldPlay2Pipeline._prepare_chunk_memory returns it); before the first
generated chunk DA3 reads the real history of the 16 fps route package. The candidates' displacements come from the package's path (poses.npy, 16 fps
like the generator: the kept video's frame f is package frame handoff_frame + f, or f without history), each candidate's delta and rho applied to it.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from . import worldplay2 as W
from .base import check_keep, relative_to_roots


def candidate_disps(poses, a, b, delta, rho):
    """[C, 3]: each candidate's displacement over the chunk from package frame a to b, in the camera at a (OpenCV: x right, y down, z forward):
    rho_i Rot(-delta_i) d_u on the ground (world z up), d_u the package path's."""
    poses = np.asarray(poses, np.float64)
    d_u = poses[b, :2, 3] - poses[a, :2, 3]
    out = []
    for dl, r in zip(delta, rho):
        t = np.deg2rad(dl)
        d = r * np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]]) @ d_u
        out.append(poses[a, :3, :3].T @ np.array([d[0], d[1], 0.0]))
    return np.array(out)


def chunk_span(chunk, keep_from):
    """(a, b): the kept video's frames a chunk moves the camera between, from the frame before its first decoded frame to its last."""
    first, last = (0, 12) if chunk == 0 else (W.CHUNK * 4 * chunk - 4, W.CHUNK * 4 * chunk + 12)
    return max(first - keep_from, 0), last - keep_from


class DepthChunks(W.CalibratedChunks):
    """CalibratedChunks without a head: each chunk with a commanded move is decided from DA3 before its first forward."""

    def __init__(self, pipe, head, *, geo, poses, kh, keep_from, **kw):
        if head is not None:
            raise ValueError("external depth: no head")
        super().__init__(pipe, None, **kw)
        self.geo, self.poses, self.kh, self.keep_from = geo, np.asarray(poses, np.float64), kh, keep_from
        self.decoded = []               # the pipeline's decoded chunks, [3, T, H, W] in [-1, 1] on the CPU

    def run(self, **kw):
        chunk = kw["chunk_start"] // W.CHUNK
        cand = self.candidates(chunk) if chunk >= self.history_chunks else None
        record, applied, began = None, False, time.perf_counter()
        if cand is not None:
            record = {"chunk": chunk, "decided": True, "choice": "KEEP"}
            rows = kw["hr_action_chunk"]
            keep = torch.tensor(cand["rows"][0], dtype=torch.float32, device=rows.device)
            check_keep(float((keep - rows.float()).abs().max()), chunk, record)
            result = self.depth_decision(chunk, cand, record)
            if result is not None:
                applied = self.apply(result, chunk, record, rows)
        x = super().run(**kw)           # no head: the plain denoising of the decided rows; it logs its own record
        if record is not None:
            plain = self.records.pop()
            record.update(applied=applied, seconds=round(time.perf_counter() - began, 3), seconds_denoise=plain.get("seconds"))
            self.records.append(record)
        return x

    def depth_decision(self, chunk, cand, record):
        from nopeworld.inference.methods.external_depth import chunk_len16, clock16, geo_choose, to_rgb
        began = time.perf_counter()
        video = torch.cat(self.decoded, dim=1)[:, self.keep_from:] if self.decoded else None       # the hand-off frame first
        n = 0 if video is None else int(video.shape[1])
        end16, index_of = clock16(n, W.FPS, first_is_handoff=True)
        length = min(chunk_len16(int(self.decoded[-1].shape[1]), W.FPS), end16) if self.decoded else 0
        self.geo.size = [W.WIDTH, W.HEIGHT] if video is None else [int(video.shape[3]), int(video.shape[2])]        # [width, height]
        a, b = chunk_span(chunk, self.keep_from)
        last = len(self.poses) - 1
        pa, pb = min(self.kh + a, last), min(self.kh + b, last)
        disps = candidate_disps(self.poses, pa, pb, cand["delta"], cand["rho"])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()           # room for DA3
        reply = self.geo.scores(chunk, end16, length, disps, lambda offs: to_rgb(video[:, [index_of(o) for o in offs]]))
        result = geo_choose(reply, self.selection)
        record.update(depth_seconds=round(time.perf_counter() - began, 2), decoded_end16=end16, no_scale=result is None,
                      package_frames=[pa, pb], past_package_end=self.kh + b > last, h_all=reply.get("h"), in_view=reply.get("in_view"),
                      depth={key: v for key, v in reply.items() if key not in ("h", "P", "A", "in_view", "slide")})
        if result is None:
            record.update(choice=self.names[0])
            return None
        result["rows"] = torch.tensor(cand["rows"], dtype=torch.float32)
        return result


def generate_case(pipe, pkg, out, *, depth, track, head=None, payload=None, observe_only=False, scorer=False, history_chunks=0, seed=None):
    """worldplay2.generate_case with DepthChunks deciding: the same video, log and refusals, plus the depth fields.
    depth: external_depth.DepthSpec; payload: the collision head file, of which only its selection (rule, tau, keep margin) is read."""
    from PIL import Image
    if track != "B" or head is not None or scorer or observe_only:
        raise ValueError("external depth: track B without a head, no async scoring or observe-only")
    pkg = Path(pkg)
    case = W.route_id(pkg)
    meta = json.loads((pkg / "meta.json").read_text())
    poses = np.load(pkg / "poses.npy")
    kh, keep_from = (meta["handoff_frame"], 4 * W.CHUNK * history_chunks - 4) if history_chunks else (0, 0)
    geo = depth.geo(case, W.crop_box(*Image.open(pkg / "handoff.png").size), i2v=not history_chunks)
    made = {}

    class Bound(DepthChunks):       # a subclass, not a factory: the pipeline's _prefill_kv_cache asks isinstance(chunks, CalibratedChunks)
        def __init__(self, pipe_, head_, **kw):
            super().__init__(pipe_, head_, geo=geo, poses=poses, kh=kh, keep_from=keep_from, **kw)
            made["chunks"] = self

    if "_prepare_chunk_memory" in vars(pipe):
        raise RuntimeError("the pipe already has its own _prepare_chunk_memory: not wrapped twice")
    prepare = pipe._prepare_chunk_memory

    def prepared(**kw):             # the pipeline's decode of the finished chunk, its first return value (on the CPU)
        result = prepare(**kw)
        if "chunks" in made:
            made["chunks"].decoded.append(result[0])
        return result

    pipe._prepare_chunk_memory = prepared
    try:
        info, kept = W.generate_case(pipe, pkg, out, track=track, head=None, payload=payload, history_chunks=history_chunks, seed=seed,
                                     chunks_class=Bound)
    finally:
        del pipe._prepare_chunk_memory
        if "chunks" in made:
            made["chunks"].decoded = []
    if "chunks" not in made:
        raise RuntimeError(f"{case}: no depth decisions were made")
    info.update(external_depth=True, depth_mode=geo.mode, depth_package=relative_to_roots(geo.package), depth_answers=geo.asked)
    (Path(out) / f"{case}.decisions.json").write_text(json.dumps(info, indent=1))
    return info, kept
