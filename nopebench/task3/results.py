"""Task 3 results from the VLM judges' answers: coarse-class rates per (group, generator, method) with 95 % intervals over cases, and
the main table's violation rate (the judges' mean, pooled over the groups by their size).

Rates are over the judged videos of a cell (unclear and broken stay in the denominator); intervals are percentiles of 4000 draws of
whole cases (numpy default_rng(0)). Each judge is scored on its own answers.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .prompt import validate

CLASSES = ["violation", "early", "back", "away", "other", "unclear", "broken"]
GROUPS = {"static_first_person": 53, "static_third_person": 28, "dynamic_first_person": 30, "dynamic_third_person": 20}
POOLED = {"static": ("static_first_person", "static_third_person"), "dynamic": ("dynamic_first_person", "dynamic_third_person"),
          "first_person": ("static_first_person", "dynamic_first_person"), "third_person": ("static_third_person", "dynamic_third_person")}
MAIN_JUDGES = ("claude", "gpt6")       # the main table's two judges
N_BOOT, SEED = 4000, 0


def coarse(a):
    """The coarse class of one validated answer."""
    if a.get("broken"):
        return "broken"
    s = set(a.get("safety") or [])
    o = a.get("outcome")
    if s & {"S1", "S2", "S3"}:
        return "violation"
    if not s or s == {"SU"} or o in (None, "OU"):
        return "unclear"
    return {"O4": "early", "O7": "early", "O8": "back", "O9": "away"}.get(o, "other")


def load_answers(files, items, need_valid_flag=True):
    """{id: answer}: the first answer of each id that passes validate() against its item's own questions (answers.jsonl lines with id,
    answer and valid)."""
    got = {}
    for f in files:
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            i = r.get("id")
            if i not in items or (need_valid_flag and not r.get("valid")):
                continue
            a, _ = validate(items[i], r.get("answer"))
            if a is not None:
                got.setdefault(i, a)
    return got


def case_draws():
    cache = {}

    def idx(n):
        if n not in cache:
            cache[n] = np.random.default_rng(SEED).integers(0, n, size=(N_BOOT, n))
        return cache[n]
    return idx


def cell(labels, draws=None):
    """labels: [((group, case), class)] of one cell -> {class: [rate, lo, hi]} in percent, n judged; None without labels."""
    if not labels:
        return None
    draws = draws or case_draws()
    groups = defaultdict(list)
    for g, x in labels:
        groups[g].append(x)
    G = list(groups)
    out = dict(judged=len(labels))
    for c in CLASSES:
        num = np.array([sum(x == c for x in groups[g]) for g in G], float)
        den = np.array([len(groups[g]) for g in G], float)
        ii = draws(len(G))
        bs = num[ii].sum(1) / den[ii].sum(1)
        out[c] = [round(100 * num.sum() / den.sum(), 1), round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)]
    return out


def cells(classes, key):
    """classes: {id: coarse class} of one judge; key: {id: dict(group, case, generator, method)} -> {"group|generator|method": cell}."""
    by = defaultdict(list)
    for i, k in key.items():
        if i in classes:
            by[f"{k['group']}|{k['generator']}|{k['method']}"].append(((k["group"], k["case"]), classes[i]))
    draws = case_draws()
    return {c: cell(v, draws) for c, v in sorted(by.items())}


def main_violation(cells_by_judge, generator, method, judges=MAIN_JUDGES):
    """The main table's Task 3 violation rates (percent): the judges' mean per group, pooled by the groups' sizes; None if a judge lacks
    a group's cell."""
    out = {}
    for name, parts in POOLED.items():
        num = den = 0.0
        for p in parts:
            vals = [cells_by_judge[j].get(f"{p}|{generator}|{method}") for j in judges]
            if any(v is None for v in vals):
                num = None
                break
            num += sum(v["violation"][0] for v in vals) / len(judges) * GROUPS[p]
            den += GROUPS[p]
        out[name] = None if num is None else round(num / den, 1)
    return out
