"""What a training or validation replay reads: rows grouped by video, which decision states are probed, and their weights."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from ..head.labels import episodes, freeze_points, is_decision, sampling_weights

ALL_SAFE = 0.05          # a state is all-safe when every candidate's collision label is at most this
NEAR_WALL = ("contact", "<1chunk", "1-3s")
HEAVY_FIELDS = ("candidate_poses", "executed_poses", "candidate_actions", "candidate_controls", "intent_controls",
                "executed_controls")


def read_jsonl(path):
    """Rows of a .jsonl or .jsonl.gz file, the large per-candidate lists as float32 arrays."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt") as fh:
        for line in fh:
            row = json.loads(line)
            for key in HEAVY_FIELDS:
                if key in row:
                    row[key] = np.asarray(row[key], dtype=np.float32)
            rows.append(row)
    return rows


def episode_seed(pair_id, seed):
    return int(hashlib.sha256(f"{seed}:{pair_id}".encode()).hexdigest()[:8], 16)


def keep_share(state_id, seed):
    """A state's fixed position in [0, 1): it is kept when this is below the kept fraction."""
    return int(hashlib.sha256(f"keep:{seed}:{state_id}".encode()).hexdigest()[:8], 16) / 2 ** 32


def all_safe(row):
    return max(row["collision_target"]) <= ALL_SAFE


def consistent(row):
    return row.get("history_consistent", True)


def probe_rule(*, all_safe_keep, seed, consistent_only=True):
    """None (probe every decision state) or the rule for which decision states get a step-0 forward."""
    if all_safe_keep >= 1.0 and not consistent_only:
        return None
    return lambda row: ((not consistent_only or consistent(row))
                        and (all_safe_keep >= 1.0 or not all_safe(row) or keep_share(row["id"], seed) < all_safe_keep))


def training_weights(population, probe, all_safe_keep):
    """(kept decision states, their weights): sampling weights over the whole population as if nothing were skipped, kept all-safe
    states times 1 / all_safe_keep, rescaled to mean 1."""
    base = sampling_weights(population)
    kept = [(r, w) for r, w in zip(population, base) if probe is None or probe(r)]
    w = np.array([w * (1.0 / all_safe_keep if all_safe_keep < 1.0 and all_safe(r) else 1.0) for r, w in kept])
    return [r for r, _ in kept], w / w.mean()


def expected_states(rows, probe):
    return sum(is_decision(r) and (probe is None or bool(probe(r))) for r in rows)


def with_states(eps, probe, frozen=False):
    """The rows of the videos that reach the heads at all: only they are prepared ahead."""
    return [ep for _, ep in eps if expected_states(ep, probe) or (frozen and freeze_points(ep))]


def spread(items, n):
    """n items evenly spread over the list, or all of them."""
    if not n or n >= len(items):
        return list(items)
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)] if n > 1 else list(items[:1])


__all__ = ["episodes", "read_jsonl", "episode_seed", "keep_share", "all_safe", "consistent", "probe_rule", "training_weights",
           "expected_states", "with_states", "spread", "NEAR_WALL"]
