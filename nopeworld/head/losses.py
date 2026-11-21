"""The training loss of one decision state.

  safe       soft BCE between sigmoid(safe logit) and the label P
  clearance  smooth L1 (beta 0.05 m) on the swept clearance, clipped to [-0.25, 1.0] m
  listwise   KL between softmax(log R / t) and softmax(log R_hat / t) over the candidates, R_hat = P_hat x A_hat
  intent     soft BCE between sigmoid(intent logit) and the slide-based intent fit

Candidates whose swept area is not visible in the history count half in the per-candidate terms; each state carries its sampling weight.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

WEIGHTS = {"safe": 1.0, "clearance": 0.5, "listwise": 1.0, "intent": 1.0}
CLEARANCE_RANGE = (-0.25, 1.0)


def listwise_kl(log_r_hat, log_r, mask, temperature):
    student = (log_r_hat / temperature).masked_fill(~mask, -torch.inf)
    teacher = (log_r / temperature).masked_fill(~mask, -torch.inf)
    log_q = torch.log_softmax(teacher, -1)
    log_p = torch.log_softmax(student, -1)
    terms = torch.where(mask, log_q.exp() * (log_q - log_p), torch.zeros_like(log_q))
    return terms.sum(-1)


def safety_losses(out, batch, *, temperature: float = 0.1, clearance_range=CLEARANCE_RANGE, weights=None):
    w = {**WEIGHTS, **(weights or {})}
    mask = batch["candidate_mask"].bool()
    z = out["safe_logits"].masked_fill(~mask, 0.0)
    state_w = batch["weight"].float()
    cand_w = (0.5 + 0.5 * batch["observed"].float()) * state_w[:, None] * mask
    denom = cand_w.sum().clamp_min(1e-8)
    safe = (F.binary_cross_entropy_with_logits(z, batch["safe_target"].float(), reduction="none") * cand_w).sum() / denom
    h = batch["clearance_target"].float().clamp(*clearance_range)
    clearance = ((F.smooth_l1_loss(out["clearance"].float(), h, beta=0.05, reduction="none") * cand_w).sum()
                 / cand_w.sum().clamp_min(1e-8))
    if "intent_logits" in out:
        y = out["intent_logits"].masked_fill(~mask, 0.0)
        intent = ((F.binary_cross_entropy_with_logits(y, batch["intent_target"].float(), reduction="none") * cand_w)
                  .sum() / denom)
        log_a = F.logsigmoid(y)
    else:
        intent, log_a = None, torch.log(batch["intent_fit"].float().clamp_min(1e-6))
    log_r_hat = F.logsigmoid(z) + log_a
    log_r = torch.log(batch["reward_target"].float().clamp_min(1e-4))
    listwise = (listwise_kl(log_r_hat, log_r, mask, temperature) * state_w).sum() / state_w.sum().clamp_min(1e-8)
    total = w["safe"] * safe + w["clearance"] * clearance + w["listwise"] * listwise
    result = {"safe": safe, "clearance": clearance, "listwise": listwise}
    if intent is not None:
        total = total + w["intent"] * intent
        result["intent"] = intent
    return {"total": total, **result}
