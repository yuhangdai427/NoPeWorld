"""The decision rule: which candidate a chunk executes, given the head's safety logits and intent scores.

safe_first: only candidates with P = sigmoid(logit / T) >= tau compete and the highest intent score wins (ties to the safer);
if none reaches tau, STOP. The command is kept unless it is itself feasible and the winner beats it by more than the keep margin.
"""
from __future__ import annotations

import torch


def choose(safe_logits, intent_fit, mask, *, keep_index=0, temperature=1.0, keep_margin=0.0, rule="product",
           tau=0.5, stop_index=1):
    p = torch.sigmoid(safe_logits.float() / temperature).masked_fill(~mask.bool(), 0.0)
    a = intent_fit.float()
    r = p * a
    if rule == "product":
        best = r.argmax(-1)
        keep = torch.as_tensor(keep_index, device=r.device).expand_as(best)
        gain = r.gather(-1, best[:, None]).squeeze(-1) - r.gather(-1, keep[:, None]).squeeze(-1)
        return {"choice": torch.where(gain > keep_margin, best, keep), "P": p, "R": r}
    if rule != "safe_first":
        raise ValueError(f"unknown rule {rule!r}")
    ok = mask.bool() & (p >= tau)
    best = torch.where(ok, a + 1e-3 * p, torch.full_like(a, -1.0)).argmax(-1)
    best = torch.where(ok.any(-1), best, torch.full_like(best, stop_index))
    keep = torch.as_tensor(keep_index, device=r.device).expand_as(best)
    keep_ok = ok.gather(-1, keep[:, None]).squeeze(-1)
    gain = a.gather(-1, best[:, None]).squeeze(-1) - a.gather(-1, keep[:, None]).squeeze(-1)
    return {"choice": torch.where(keep_ok & (gain <= keep_margin), keep, best), "P": p, "R": r}
