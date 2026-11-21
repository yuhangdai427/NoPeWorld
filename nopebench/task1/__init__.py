"""Task 1: the heads' decisions on the true history (teacher-forced replay), every candidate labelled by the scene's occupancy.

    missed   % of unsafe inputs that are kept (KEEP's P(safe) label < 0.5)
    false    % of safe inputs that are changed
    score    100 (1 - mean regret), regret = max_i R_i - R_sent with R = P x A (safety x intent), the training reward of a candidate
    raw      the generator alone keeps every input: missed 100, false 0, score from the regret of always keeping
Prediction mode makes no decision at a route's first replayed chunk, so a generator with both modes is scored on the decisions both make
(`common`); `own` scores each mode on all its decisions.

Report (task1_report.json): {"heads": {name: {"mode", "per_state": [{"video", "chunk", "choice", "keep_safe", "regret"}],
"always_keep": {"regret"}}}}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nopeworld.head.candidates import NAMES

REPORT = "task1_report.json"


def mode_of(name, head):
    """A head's mode: its own `mode`, else from its name (a head that reads the commit pass is prediction mode)."""
    return head.get("mode") or ("prediction" if name.endswith("_commit") else "collision")


def stats(states):
    """Missed / false / score of a list of per-state decisions (missed or false is None when there is no unsafe or safe input)."""
    uns = [x for x in states if not x["keep_safe"]]
    saf = [x for x in states if x["keep_safe"]]
    share = lambda hits, of: round(100 * hits / len(of), 1) if of else None
    return dict(decisions=len(states), missed=share(sum(x["choice"] == "KEEP" for x in uns), uns),
                false=share(sum(x["choice"] != "KEEP" for x in saf), saf),
                score=round(100 * (1 - sum(x["regret"] for x in states) / len(states)), 1))


def per_state_from_rows(rows, decisions):
    """per_state entries from Task 1 rows and decisions {(pair_id, chunk): candidate name}: keep_safe = KEEP's P(safe) >= 0.5, regret =
    max_i R_i - R_choice with R the row's policy_target. Returns (per_state, always_keep regret)."""
    out, keep_regret = [], []
    for row in rows:
        key = (row["pair_id"], int(row["chunk"]))
        if key not in decisions:
            continue
        reward = np.asarray(row["policy_target"], np.float32).astype(float)
        safe = (1.0 - np.asarray(row["collision_target"], np.float32)) >= 0.5
        roles = list(row.get("candidate_roles") or NAMES)
        k = roles.index(decisions[key])
        best = int(reward.argmax())
        out.append(dict(video=key[0], chunk=key[1], choice=roles[k], keep_safe=bool(safe[0]), best=roles[best], regret=round(float(reward[best] - reward[k]), 4)))
        keep_regret.append(float(reward[best] - reward[0]))
    return out, float(np.mean(keep_regret)) if keep_regret else None


def table_row(generator, heads):
    """One generator's Task 1 row from its heads' reports {name: head report}."""
    names = {mode_of(name, h): name for name, h in heads.items()}
    per = {mode: {(x["video"], x["chunk"]): x for x in heads[name]["per_state"]} for mode, name in names.items()}
    common = set.intersection(*(set(v) for v in per.values()))
    last = "prediction" if "prediction" in names else "collision"
    if set(per[last]) != common:
        raise ValueError(f"{generator}: the decisions of {last} mode are not the common ones; the raw score needs per-decision keep regret")
    raw = round(100 * (1 - heads[names[last]]["always_keep"]["regret"]), 1)
    row = dict(generator=generator, decisions=len(common), raw=dict(missed=100.0, false=0.0, score=raw), modes={})
    for mode, name in names.items():
        row["modes"][mode] = dict(head=name, common=stats([per[mode][k] for k in sorted(common)]), own=stats(list(per[mode].values())))
    return row


def evaluate_run(runs, generator):
    """The Task 1 row of the heads in one or more run folders' reports."""
    heads = {}
    for run in [runs] if isinstance(runs, (str, Path)) else runs:
        for name, head in json.loads((Path(run) / REPORT).read_text())["heads"].items():
            if name in heads:
                raise ValueError(f"head {name} is in more than one run folder")
            heads[name] = head
    return table_row(generator, heads)
