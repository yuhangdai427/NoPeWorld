"""Targets and weights of the training rows: one decision state per row, labels from the scene's occupancy.

Only decision states train the head: chunks where the command moves (intent_length_m > 0). Each state's loss is scaled by a
sampling weight (mean 1) that sets the time-to-contact mix (>3 s, 1-3 s, <1 chunk, contact = 30/30/30/10, near-wall clips 0.15)
and gives every chunk index the same collision share.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

import numpy as np
import torch

from .candidates import W_LATERAL, intent_fit

STRATUM_MIX = {">3s": 0.3, "1-3s": 0.3, "<1chunk": 0.3, "contact": 0.1, "nearwall": 0.15}
FAR_FROM_WALL = 0.15


def is_decision(row) -> bool:
    """A chunk where the command moves; `decision` False marks a first chunk that cannot be decided."""
    return row.get("decision", True) and row.get("intent_length_m", 0.0) > 1e-3


def episodes(rows, order=None):
    """Rows grouped per video, chunks in order; videos in file order, or in `order` (ids) and only those."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["pair_id"]].append(row)
    ids = list(groups) if order is None else [i for i in order if i in groups]
    return [(pid, sorted(groups[pid], key=lambda r: r["chunk"])) for pid in ids]


def freeze_points(rows, *, seed=202):
    """The chunks of one video where a replay also reads the state after one frozen (held) chunk, chosen from the labels and a hash
    of the state id: the first unsafe command (or the state nearest a wall) and one safe state far from any wall; one for a near-wall clip."""
    decisions = [r for r in rows if is_decision(r) and r.get("history_consistent", True)]
    if not decisions:
        return []
    last = max(r["chunk"] for r in decisions)
    pool = [r for r in decisions if 0 < r["chunk"] < last]
    by_hash = sorted(pool, key=lambda r: hashlib.sha256(f"freeze:{seed}:{r['id']}".encode()).hexdigest())
    if not pool:
        return []
    if any(r.get("stratum") == "nearwall" for r in decisions):
        return [by_hash[0]["chunk"]]
    unsafe = lambda r: r["collision_target"][r["keep_index"]] > 0.5
    clearance = lambda r: r["candidate_rim_clearance_m"][1]
    by_chunk = sorted(pool, key=lambda r: r["chunk"])
    near = next((r for r in by_chunk if unsafe(r)), None) or min(by_chunk, key=clearance)
    far = next((r for r in by_hash if r is not near and not unsafe(r) and clearance(r) >= FAR_FROM_WALL), None)
    return sorted(r["chunk"] for r in (near, far) if r is not None)


def sampling_weights(rows, *, mix=STRATUM_MIX, clip=(0.1, 10.0)):
    n = len(rows)
    freq = Counter(r.get("stratum") for r in rows)
    w = np.array([mix.get(r.get("stratum"), 1.0 / max(len(mix), 1)) / (freq[r.get("stratum")] / n) for r in rows])
    collide = np.array([r.get("gt_status") == "collide" for r in rows])
    overall = float((w * collide).sum() / w.sum())
    by_chunk = defaultdict(list)
    for i, r in enumerate(rows):
        by_chunk[r["chunk"]].append(i)
    for idx in by_chunk.values():
        idx = np.array(idx)
        rate = float((w[idx] * collide[idx]).sum() / w[idx].sum())
        if 0.0 < rate < 1.0:
            w[idx] *= np.where(collide[idx], overall / rate, (1 - overall) / (1 - rate))
    w = np.clip(w / w.mean(), *clip)
    return w / w.mean()


def row_targets(row, *, w=W_LATERAL):
    for key in ("candidate_delta_deg", "candidate_rho", "intent_length_m"):
        if key not in row:
            raise KeyError(f"row {row['id']} has no {key}")
    targets = {"safe_target": 1.0 - torch.tensor(row["collision_target"], dtype=torch.float32),
               "reward_target": torch.tensor(row["policy_target"], dtype=torch.float32),
               "clearance_target": torch.tensor(row["candidate_rim_clearance_m"], dtype=torch.float32),
               "observed": torch.tensor(row["candidate_observed"], dtype=torch.float32),
               "intent_fit": intent_fit(row["candidate_delta_deg"], row["candidate_rho"], w=w)}
    if "candidate_A_slide" in row:
        targets["intent_target"] = torch.tensor(row["candidate_A_slide"], dtype=torch.float32)
    return targets


def row_batch(row, *, w=W_LATERAL, device=None):
    """The labels of one decision state as a batch of one."""
    targets = {k: v[None].to(device) for k, v in row_targets(row, w=w).items()}
    n = targets["safe_target"].shape[1]
    mask = row.get("candidate_mask")
    mask = torch.ones(1, n, dtype=torch.bool, device=device) if mask is None else torch.tensor(
        [mask], dtype=torch.bool, device=device)
    return {"candidate_mask": mask,
            "weight": torch.ones(1, device=device),
            "keep_index": torch.tensor([row["keep_index"]], device=device), **targets}
