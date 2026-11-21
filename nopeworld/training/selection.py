"""The operating point of a head, chosen on validation states only.

Temperature T by the soft negative log-likelihood of the labels; then, over a grid of tau and keep margin delta, the lowest regret
among the points whose missed-intervention rate is at most the mode's limit (or the lowest missed rate if none reaches it).

    regret   R of the labels' best candidate minus R of the chosen one, the mean over states
    missed   the command is unsafe and the head kept it, among states with an unsafe command
    false    the command is safe and the head changed it, among states with a safe command
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..head.choose import choose

TEMPERATURES = np.round(np.exp(np.linspace(np.log(0.5), np.log(4.0), 22)), 3)
MARGINS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2)
TAUS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
RATES = {"regret": ("regret", "states"), "missed": ("missed", "command_unsafe"), "false": ("false", "command_safe"),
         "wrong_change": ("wrong", "changed")}


def auc(score, positive):
    """ROC AUC, ties counted half."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    if positive.all() or not positive.any():
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_score = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_score[j + 1] == sorted_score[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def selection_metrics(choice, reward, safe, keep_index=0):
    """choice [S]; reward, safe [S, N] labels."""
    choice, reward, safe = np.asarray(choice), np.asarray(reward, float), np.asarray(safe, float)
    rows = np.arange(len(choice))
    best = reward.argmax(1)
    keep_best, keep_unsafe = best == keep_index, safe[:, keep_index] < 0.5
    chose_keep = choice == keep_index
    return {"states": int(len(choice)),
            "regret": float((reward[rows, best] - reward[rows, choice]).mean()),
            "agree_argmax": float((choice == best).mean()),
            "unsafe_choice": float((safe[rows, choice] < 0.5).mean()),
            "false_intervention": float((~chose_keep[keep_best]).mean()) if keep_best.any() else None,
            "missed_intervention": float(chose_keep[keep_unsafe].mean()) if keep_unsafe.any() else None}


def evaluate(pred, *, temperature=1.0, keep_margin=0.0, rule="product", tau=0.5):
    """pred: per-state stacks of logits, intent_fit, candidate_mask, keep_index, reward_target, safe_target."""
    if bool((pred["keep_index"] != 0).any()):
        raise ValueError("selection_metrics assumes KEEP is candidate 0")
    sel = choose(pred["logits"], pred["intent_fit"], pred["candidate_mask"], keep_index=pred["keep_index"],
                 temperature=temperature, keep_margin=keep_margin, rule=rule, tau=tau)
    metrics = selection_metrics(sel["choice"].numpy(), pred["reward_target"].numpy(), pred["safe_target"].numpy())
    keep = pred["keep_index"][:, None]
    keep_unsafe = pred["safe_target"].gather(1, keep).squeeze(1).numpy() < 0.5
    metrics["keep_unsafe_auc"] = auc(1 - sel["P"].gather(1, keep).squeeze(1).numpy(), keep_unsafe)
    return metrics


def soft_nll(pred, temperature):
    mask = pred["candidate_mask"]
    p = torch.sigmoid(pred["logits"][mask] / temperature).clamp(1e-6, 1 - 1e-6)
    y = pred["safe_target"][mask]
    return float(-(y * p.log() + (1 - y) * (1 - p).log()).mean())


def some_states(pred, keep):
    keep_t = torch.as_tensor(keep)
    out = {k: pred[k][keep_t] for k in ("logits", "intent_fit", "candidate_mask", "keep_index", "reward_target", "safe_target")}
    out.update({k: np.asarray(pred[k])[keep] for k in ("chunk", "video", "stratum")})
    return out


def states_with_a_read(pred, state):
    """A prediction head has no state before the first generated chunk, where it keeps the command: those states are left out."""
    if state != "commit":
        return pred
    keep = pred["has_state"] if "has_state" in pred else np.asarray(pred["chunk"]) > 0
    return some_states(pred, keep)


def outcomes(pred, **point):
    if bool((pred["keep_index"] != 0).any()):
        raise ValueError("KEEP must be candidate 0")
    choice = choose(pred["logits"], pred["intent_fit"], pred["candidate_mask"], keep_index=pred["keep_index"], rule="safe_first",
                    **point)["choice"].numpy()
    reward, safe = pred["reward_target"].numpy().astype(float), pred["safe_target"].numpy().astype(float)
    rows = np.arange(len(choice))
    changed, unsafe = choice != 0, safe[:, 0] < 0.5
    return {"regret": reward[rows, reward.argmax(1)] - reward[rows, choice], "states": np.ones(len(choice)),
            "command_unsafe": unsafe, "missed": ~changed & unsafe, "command_safe": ~unsafe, "false": changed & ~unsafe,
            "changed": changed, "wrong": changed & (safe[rows, choice] < 0.5)}


def rate(out, name):
    top, bottom = RATES[name]
    total = float(np.sum(out[bottom]))
    return float(np.sum(out[top])) / total if total else None


def working_point(pred, limit):
    """(operating point {temperature, tau, keep_margin}, whether its missed rate is within the limit)."""
    temperature = float(min(TEMPERATURES, key=lambda t: soft_nll(pred, t)))
    points = []
    for tau in TAUS:
        for margin in MARGINS:
            point = dict(temperature=temperature, tau=float(tau), keep_margin=float(margin))
            out = outcomes(pred, **point)
            points.append((point, rate(out, "regret"), rate(out, "missed")))
    within = [p for p in points if p[2] is None or p[2] <= limit]
    if within:
        return min(within, key=lambda p: p[1])[0], True
    return min(points, key=lambda p: (p[2], p[1]))[0], False


def select_head(run, name, *, state, limit, generator, mode, out):
    """The operating point of head `name` of a finished training run, written with the head as `out` (a released head.pt)."""
    run = Path(run)
    predictions = torch.load(run / "val_preds.pt", map_location="cpu", weights_only=False)
    pred = states_with_a_read(predictions[name], state)
    point, within = working_point(pred, limit)
    o = outcomes(pred, **point)
    payload = torch.load(run / f"head_{name}.pt", map_location="cpu", weights_only=False)
    payload.update(selection=dict(rule="safe_first", **point), generator=generator, mode=mode)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    summary = {"head": name, "mode": mode, "missed_limit": limit, "within_limit": within, "point": point, "states": int(len(o["states"])),
               "metrics": {k: rate(o, k) for k in RATES}}
    (out.parent / "selection.json").write_text(json.dumps(summary, indent=1))
    return summary
