"""Compares two per-state loss logs (losses.jsonl) head by head, in the order the states were learned from.

    python tests/gpu/compare_losses.py <reference.jsonl> <candidate.jsonl> [--tolerance 0] [--out result.json]

Per optimizer step it reports the mean total loss of the states of that step on both sides and the largest difference of any term.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

TERMS = ("total", "safe", "clearance", "listwise", "intent")


def read(path):
    by_head = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_head[r["head"]].append(r)
    return by_head


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--tolerance", type=float, default=0.0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    ref, cand = read(a.reference), read(a.candidate)
    result, ok = {"heads": {}}, True
    for head in sorted(set(ref) | set(cand)):
        r, n = ref.get(head, []), cand.get(head, [])
        count = min(len(r), len(n))
        worst = max((abs(x[t] - y[t]) for x, y in zip(r[:count], n[:count]) for t in TERMS if t in x and t in y), default=0.0)
        first = next((i for i, (x, y) in enumerate(zip(r, n)) if any(abs(x[t] - y[t]) > a.tolerance for t in TERMS if t in x and t in y)), None)
        steps = defaultdict(lambda: [[], []])
        for side, rows in ((0, r[:count]), (1, n[:count])):
            for row in rows:
                steps[row["step"]][side].append(row["total"])
        per_step = [{"step": s, "states": len(v[0]), "reference": sum(v[0]) / len(v[0]), "candidate": sum(v[1]) / len(v[1])}
                    for s, v in sorted(steps.items()) if v[0] and v[1]]
        same = len(r) == len(n) and first is None
        ok = ok and same
        result["heads"][head] = {"states": [len(r), len(n)], "optimizer_steps": [max((x["step"] for x in r), default=-1) + 1, max((x["step"] for x in n), default=-1) + 1],
                                 "max_abs_difference": worst, "first_state_over_tolerance": first, "equal": same, "per_step_mean_total": per_step}
    result["equal"] = ok
    text = json.dumps(result, indent=1)
    if a.out:
        a.out.write_text(text)
    print(text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
