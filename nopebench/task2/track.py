"""Obstacle tracking with SAM 2.1 from the instance mask at the hand-off frame (a real render), forward through the generated frames.

SAM 2 is not shipped: install its package (sam2) and put the checkpoint under NOPE_GENERATORS/sam2/ (or set NOPE_SAM2_CHECKPOINT).
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np

from nopeworld.paths import generators_path

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
_PREDICTOR = []


def checkpoint():
    return os.environ.get("NOPE_SAM2_CHECKPOINT") or str(generators_path("sam2", "sam2.1_hiera_large.pt"))


def predictor(device="cuda"):
    if not _PREDICTOR:
        from sam2.build_sam import build_sam2_video_predictor
        _PREDICTOR.append(build_sam2_video_predictor(SAM2_CONFIG, checkpoint(), device=device))
    return _PREDICTOR[0]


def track(frames_bgr, handoff, mask, *, width=640):
    """frames_bgr: all video frames; mask: bool H x W at the hand-off frame. Returns the masks at the frames' resolution and SAM 2's
    object-presence logits for frames handoff..end."""
    import cv2
    import torch
    h, w = frames_bgr[0].shape[:2]
    scale = width / w
    size = (width, int(round(h * scale)))
    pred = predictor()
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames_bgr[handoff:]):
            cv2.imwrite(str(Path(tmp) / f"{i:05d}.jpg"), cv2.resize(f, size), [cv2.IMWRITE_JPEG_QUALITY, 95])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = pred.init_state(video_path=tmp)
            small = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
            pred.add_new_mask(state, frame_idx=0, obj_id=1, mask=small)
            masks, scores = {}, {}
            for idx, ids, logits in pred.propagate_in_video(state):
                m = (logits[0, 0] > 0).float().cpu().numpy()
                masks[idx] = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                scores[idx] = float(logits[0, 0].max())
    n = len(frames_bgr) - handoff
    return [masks.get(i, np.zeros((h, w), bool)) for i in range(n)], np.array([scores.get(i, -1e3) for i in range(n)])
