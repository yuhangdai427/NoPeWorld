"""Online training of the safety heads while the frozen generator replays the training videos.

Each video is replayed once, teacher-forced: at every probed decision state the tapped block tokens and each candidate's camera go to
observe(), where every head first scores the state with its EMA weights (prequential statistics) and then takes one forward/backward
step; no hidden state is stored. The EMA weights are what is validated and kept; selection.py then fixes each head's operating point.
Heads with the same layers and replica start from the same random weights; prediction heads and heads with `frozen` draw their dropout
from a generator of their own, so the other heads train the same with or without them.
"""
from __future__ import annotations

from collections import defaultdict
import contextlib
import copy
import dataclasses
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

from ..head.candidates import NAMES, SIGMA_A, W_LATERAL
from ..head.choose import choose
from ..head.labels import episodes, is_decision, row_batch
from ..head.losses import safety_losses
from ..head.model import FieldConfig, SafetyPolicy
from .replay import (consistent, episode_seed, expected_states, probe_rule, read_jsonl, spread, training_weights,
                     with_states)
from .selection import evaluate, selection_metrics, some_states


@dataclass(frozen=True)
class HeadSpec:
    """A head: the blocks it reads and how it trains. Written "<blocks>[:option,...]", e.g. "27", "30+38+39:acc32,lr2", "20:commit"."""
    layers: tuple[int, ...]
    init: str = "default"
    clearance: str = "plain"
    replica: int = 0
    accumulate: int = 0              # states per update; 0 = the run's accumulate
    lr_scale: int = 1
    query: str = "shared"
    state: str = "step0"             # step0: collision head | commit: prediction head (reads the previous chunk's commit pass)
    arm: str = ""
    frozen: bool = False             # also learns from the state after one held chunk

    @property
    def name(self) -> str:
        parts = ["L" + "+".join(map(str, self.layers))]
        if self.replica:
            parts.append(f"r{self.replica}")
        if self.accumulate:
            parts.append(f"acc{self.accumulate}")
        if self.lr_scale != 1:
            parts.append(f"lr{self.lr_scale}")
        if self.state != "step0":
            parts.append("commit")
        if self.frozen:
            parts.append("frozen")
        return "_".join(parts)

    def to_dict(self) -> dict:
        options = {k: v for k, v in asdict(self).items() if k != "frozen" or v}
        return {**options, "layers": list(self.layers), "name": self.name}


def parse_head(text: str) -> HeadSpec:
    layers, _, options = text.partition(":")
    spec = {"layers": tuple(int(x) for x in layers.split("+"))}
    for option in filter(None, options.split(",")):
        if option.startswith("r") and option[1:].isdigit():
            spec["replica"] = int(option[1:])
        elif option.startswith("acc") and option[3:].isdigit():
            spec["accumulate"] = int(option[3:])
        elif option.startswith("lr") and option[2:].isdigit():
            spec["lr_scale"] = int(option[2:])
        elif option == "commit":
            spec["state"] = "commit"
        elif option == "frozen":
            spec["frozen"] = True
        else:
            raise ValueError(f"unknown head option {option!r} in {text!r}")
    return HeadSpec(**spec)


def spec_of(head_config: dict) -> HeadSpec:
    """The HeadSpec of a generator config's head entry (blocks, state, frozen, accumulate, lr_scale)."""
    return HeadSpec(layers=tuple(head_config["blocks"]), state=head_config.get("state", "step0"), frozen=bool(head_config.get("frozen", False)),
                    accumulate=int(head_config.get("accumulate", 0)), lr_scale=int(head_config.get("lr_scale", 1)))


def init_seed(spec: HeadSpec, seed: int) -> int:
    """The random-initialisation seed of a head: its layers and replica only."""
    key = f"{seed}:{'+'.join(map(str, spec.layers))}:r{spec.replica}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def intent_of(out, batch):
    return torch.sigmoid(out["intent_logits"].float()) if "intent_logits" in out else batch["intent_fit"]


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1))
    tmp.replace(path)


def json_ready(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def atomic_save(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def check_camera(camera, row):
    if not torch.isfinite(camera).all():
        raise FloatingPointError(f"{row['id']}: invalid candidate camera")


class Head:
    """One SafetyPolicy: online model, optimizer, EMA copy and the round's prequential statistics."""

    def __init__(self, spec, all_layers, *, lr, warmup, ema_decay, device, base=FieldConfig(), commit_base=None, seed=202, accumulate=16):
        self.spec = spec
        self.max_steps = None
        if self.spec.state == "commit" and commit_base is not None:
            base = commit_base
        self.name, self.layers = self.spec.name, self.spec.layers
        self.index = [all_layers.index(l) for l in self.layers]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed(self.spec, seed))
            model = SafetyPolicy(dataclasses.replace(base, layers=self.layers, query_mode=self.spec.query, command_input=self.spec.state == "commit"))
        self.model = model.to(device)
        self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.accumulate = self.spec.accumulate or accumulate
        scale = self.accumulate / accumulate
        warmup_steps = max(1, round(warmup / scale))
        self.ema_decay = ema_decay ** scale
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr * self.spec.lr_scale, weight_decay=0.05)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda s: min(1.0, (s + 1) / warmup_steps))
        self.pending = self.steps = 0
        self.round = defaultdict(float)

    def take(self, hidden):
        return hidden[self.index][None]

    @property
    def capped(self):
        return self.max_steps is not None and self.steps >= self.max_steps and not self.pending

    def score(self, model, state, cam, ends, batch):
        return model(self.take(state), cam, ends, batch["candidate_mask"])

    def step(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for e, p in zip(self.ema.parameters(), self.model.parameters()):
                e.lerp_(p, 1 - self.ema_decay)
            for e, b in zip(self.ema.buffers(), self.model.buffers()):
                e.copy_(b)
        self.pending = 0
        self.steps += 1

    def state_dict(self):
        grads = {n: p.grad.detach().cpu() for n, p in self.model.named_parameters() if p.grad is not None}
        return {"model": self.model.state_dict(), "ema": self.ema.state_dict(), "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(), "pending": self.pending, "steps": self.steps,
                "grads": grads, "round": dict(self.round)}

    def load_state_dict(self, state):
        for key in ("model", "ema", "optimizer", "scheduler"):
            getattr(self, key).load_state_dict(state[key])
        for name, param in self.model.named_parameters():
            if name in state.get("grads", {}):
                param.grad = state["grads"][name].to(param.device)
        self.pending, self.steps = state["pending"], state["steps"]
        self.round = defaultdict(float, state.get("round", {}))


@contextlib.contextmanager
def own_rng(head, device, row, read=""):
    """A prediction head's or a `frozen` head's dropout on a generator of its own, by head and state."""
    if head.spec.state != "commit" and not head.spec.frozen:
        yield
        return
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(int(hashlib.sha256(f"{head.name}:{row['id']}{read}".encode()).hexdigest()[:8], 16))
        yield


class MultiHeadTrainer:
    def __init__(self, specs, *, accumulate=16, lr=3e-4, warmup=30, ema_decay=0.99, w=W_LATERAL, device="cuda",
                 base=FieldConfig(), commit_base=None, seed=202, loss_log=None):
        if len({s.name for s in specs}) != len(specs):
            raise ValueError("head names must be unique")
        self.all_layers = tuple(sorted({l for s in specs for l in s.layers}))
        self.heads = [Head(s, self.all_layers, lr=lr, warmup=warmup, ema_decay=ema_decay, device=device, base=base, commit_base=commit_base,
                           seed=seed, accumulate=accumulate) for s in specs]
        self.accumulate, self.w, self.device = accumulate, w, device
        self.commit_state = any(s.state == "commit" for s in specs)
        self.rule = {"rule": "product", "tau": 0.5}
        self.seen = self.frozen_seen = 0
        self.base_round = defaultdict(float)
        self.last_round_baseline = None
        self.loss_log = loss_log

    @property
    def steps(self):
        return self.heads[0].steps

    def all_capped(self):
        return all(h.capped for h in self.heads)

    def log_loss(self, head, row, losses, read=""):
        if self.loss_log is not None:
            self.loss_log.write(json.dumps({"head": head.name, "state": row["id"] + read, "step": head.steps, "pending": head.pending,
                                            **{k: float(v.detach()) for k, v in losses.items()}}) + "\n")

    def observe(self, hidden, camera, interval_ends, row, *, weight=1.0, commit_hidden=None):
        """hidden [layers, F, H, W, D] at step 0; commit_hidden the previous chunk's commit pass (None at the first chunk)."""
        check_camera(camera, row)
        batch = row_batch(row, w=self.w, device=hidden.device)
        cam, ends = camera[None], torch.as_tensor(interval_ends, dtype=torch.float32, device=hidden.device)[None]
        reward, safe = batch["reward_target"][0].cpu(), batch["safe_target"][0].cpu()
        keep_unsafe, keep_best = bool(safe[0] < 0.5), bool(int(reward.argmax()) == 0)
        base = self.base_round
        base["weight"] += weight
        base["regret"] += weight * float(reward.max() - reward[0])
        base["states"] += 1
        for head in self.heads:
            state = commit_hidden if head.spec.state == "commit" else hidden
            if state is None:
                continue
            with own_rng(head, state.device, row):
                self.observe_head(head, state, row, batch, cam, ends, weight=weight, keep_unsafe=keep_unsafe, keep_best=keep_best)
        self.seen += 1

    def observe_head(self, head, state, row, batch, cam, ends, *, weight, keep_unsafe, keep_best):
        with torch.no_grad(), torch.autocast(device_type=state.device.type, enabled=False):
            seen = head.score(head.ema, state, cam, ends, batch)
        sel = choose(seen["safe_logits"], intent_of(seen, batch), batch["candidate_mask"], **self.rule)
        m = selection_metrics(sel["choice"].cpu().numpy(), batch["reward_target"].cpu().numpy(), batch["safe_target"].cpu().numpy())
        chose_keep = int(sel["choice"][0]) == 0
        stats = head.round
        stats["weight"] += weight
        stats["regret"] += weight * m["regret"]
        stats["must"] += weight * keep_unsafe
        stats["missed"] += weight * (keep_unsafe and chose_keep)
        stats["keep_best"] += weight * keep_best
        stats["false"] += weight * (keep_best and not chose_keep)
        stats["states"] += 1
        if head.capped:
            return
        head.model.train()
        with torch.enable_grad(), torch.autocast(device_type=state.device.type, enabled=False):
            out = head.score(head.model, state, cam, ends, batch)
            losses = safety_losses(out, batch)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"{row['id']}: head {head.name}: loss {float(losses['total'])}")
            (losses["total"] * weight / head.accumulate).backward()
        self.log_loss(head, row, losses)
        stats["loss"] += weight * float(losses["total"].detach())
        head.pending += 1
        if head.pending >= head.accumulate:
            head.step()

    def observe_frozen(self, camera, interval_ends, row, *, commit_hidden, hidden=None, weight=1.0):
        """The state after one held chunk: only heads with `frozen` learn from it, at weight 1."""
        check_camera(camera, row)
        device = (commit_hidden if commit_hidden is not None else hidden).device
        batch = row_batch(row, w=self.w, device=device)
        cam, ends = camera[None], torch.as_tensor(interval_ends, dtype=torch.float32, device=device)[None]
        for head in self.heads:
            if not head.spec.frozen or head.capped:
                continue
            state = commit_hidden if head.spec.state == "commit" else hidden
            if state is None:
                continue
            with own_rng(head, state.device, row, read=":frozen"):
                head.model.train()
                with torch.enable_grad(), torch.autocast(device_type=state.device.type, enabled=False):
                    losses = safety_losses(head.score(head.model, state, cam, ends, batch), batch)
                    if not torch.isfinite(losses["total"]):
                        raise FloatingPointError(f"{row['id']}: head {head.name}: loss {float(losses['total'])} after a held chunk")
                    (losses["total"] * weight / head.accumulate).backward()
                self.log_loss(head, row, losses, read=":frozen")
                head.pending += 1
                if head.pending >= head.accumulate:
                    head.step()
        self.frozen_seen += 1

    def step(self):
        for head in self.heads:
            if head.pending:
                head.step()

    def finish_round(self):
        b, bw = self.base_round, max(self.base_round["weight"], 1e-9)
        self.last_round_baseline = {"regret": b["regret"] / bw, "weight": b["weight"], "states": int(b["states"])}
        out = {}
        for head in self.heads:
            s, w = head.round, max(head.round["weight"], 1e-9)
            out[head.name] = {"regret": s["regret"] / w, "loss": s["loss"] / w, "weight": s["weight"],
                              "gain_vs_keep": self.last_round_baseline["regret"] - s["regret"] / w,
                              "missed_intervention": s["missed"] / s["must"] if s["must"] else None,
                              "false_intervention": s["false"] / s["keep_best"] if s["keep_best"] else None,
                              "states": int(s["states"])}
            head.round = defaultdict(float)
        self.base_round = defaultdict(float)
        return out

    def state_dict(self):
        return {"heads": [h.state_dict() for h in self.heads], "names": [h.name for h in self.heads], "seen": self.seen,
                "frozen_seen": self.frozen_seen, "base_round": dict(self.base_round)}

    def load_state_dict(self, state):
        names = [h.name for h in self.heads]
        if state.get("names") != names:
            raise ValueError(f"checkpoint heads {state.get('names')} != this run's heads {names}")
        for h, s in zip(self.heads, state["heads"]):
            h.load_state_dict(s)
        self.seen, self.frozen_seen = state["seen"], state.get("frozen_seen", 0)
        self.base_round = defaultdict(float, state.get("base_round", {}))


class Collector:
    """Per-state logits and labels of a replay without gradients, for every head's EMA model."""

    def __init__(self, trainer, heads=None):
        self.trainer = trainer
        self.heads = trainer.heads if heads is None else heads
        self.out = {h.name: defaultdict(list) for h in self.heads}

    def read(self, head, state, row, batch, cam, ends):
        o = self.out[head.name]
        if state is None:
            logits, intent = torch.full_like(batch["safe_target"], -6.9), batch["intent_fit"]
            logits[:, 0] = 6.9
        else:
            with torch.autocast(device_type=state.device.type, enabled=False):
                out = head.score(head.ema, state, cam, ends, batch)
            logits, intent = out["safe_logits"], intent_of(out, batch)
        o["has_state"].append(state is not None)
        o["logits"].append(logits.float().cpu())
        o["intent_fit"].append(intent.float().cpu())
        for key in ("safe_target", "reward_target", "candidate_mask", "keep_index"):
            o[key].append(batch[key].cpu())
        o["chunk"].append(row["chunk"])
        o["video"].append(row.get("pair_id", row["id"]))
        o["stratum"].append(row.get("stratum"))
        o["id"].append(row["id"])

    @torch.no_grad()
    def observe(self, hidden, camera, interval_ends, row, *, weight=1.0, commit_hidden=None):
        check_camera(camera, row)
        batch = row_batch(row, w=self.trainer.w, device=hidden.device)
        cam, ends = camera[None], torch.as_tensor(interval_ends, dtype=torch.float32, device=hidden.device)[None]
        for head in self.heads:
            state = commit_hidden if head.spec.state == "commit" else hidden
            self.read(head, state, row, batch, cam, ends)

    @torch.no_grad()
    def observe_frozen(self, camera, interval_ends, row, *, commit_hidden, hidden=None, weight=1.0):
        check_camera(camera, row)
        device = (commit_hidden if commit_hidden is not None else hidden).device
        batch = row_batch(row, w=self.trainer.w, device=device)
        cam, ends = camera[None], torch.as_tensor(interval_ends, dtype=torch.float32, device=device)[None]
        for head in self.heads:
            state = commit_hidden if head.spec.state == "commit" else hidden
            if state is not None:
                self.read(head, state, row, batch, cam, ends)

    def predictions(self):
        result = {}
        for name, o in self.out.items():
            pred = {k: torch.cat(v) for k, v in o.items() if k not in ("chunk", "video", "stratum", "has_state", "id")}
            pred["has_state"] = np.array(o["has_state"], bool)
            pred["chunk"] = np.array(o["chunk"])
            pred["video"], pred["stratum"] = np.array(o["video"]), np.array(o["stratum"])
            pred["id"] = np.array(o["id"])
            result[name] = pred
        return result


@dataclass
class TrainSettings:
    """The trainer's settings; a generator config's `training:` block overrides the defaults."""
    rows_train: Path
    stream: Path
    rows_val: Path
    rows_val_extra: Path | None
    output: Path
    max_rounds: int = 48
    max_steps: int | None = 1000
    accumulate: int = 8
    lr: float = 3e-4
    warmup: int = 30
    ema_decay: float = 0.99
    w: float = W_LATERAL
    seed: int = 202
    all_safe_keep: float = 0.333
    rule: str = "safe_first"
    tau: float = 0.8
    intent_head: bool = True
    keep_activations: bool = True
    final_val_videos: int = 15
    frozen_states: bool = False
    limit_train: int | None = None
    limit_val: int | None = None
    checkpoint_every: int = 20
    prefetch: int = 1
    threads: int = 16


def episode_order(rows_path, stream, max_rounds, limit):
    plan = [v for v in stream["videos"] if v["round"] < max_rounds][:limit]
    eps = episodes(read_jsonl(rows_path), [v["id"] for v in plan])
    missing = [v["id"] for v in plan if v["id"] not in {pid for pid, _ in eps}]
    if missing:
        raise ValueError(f"{len(missing)} stream videos have no rows in {rows_path}: {missing[:5]}")
    return plan, eps


def train(settings: TrainSettings, adapter, specs, generator_id):
    """Trains the heads `specs` on one replay of the stream; writes head_<name>.pt, val_preds.pt, report.json, losses.jsonl."""
    s = settings
    torch.set_num_threads(s.threads)
    output = Path(s.output)
    output.mkdir(parents=True, exist_ok=True)
    stream = json.loads(Path(s.stream).read_text())
    plan, train_eps = episode_order(s.rows_train, stream, s.max_rounds, s.limit_train)
    round_of = {v["id"]: v["round"] for v in plan}
    val_eps = [(pid, ep) for pid, ep in episodes(read_jsonl(s.rows_val)) if ep[0]["view"] in stream["views"]]
    val_eps = spread(val_eps, s.limit_val)
    val_extra = episodes(read_jsonl(s.rows_val_extra)) if s.rows_val_extra else []
    probe = probe_rule(all_safe_keep=s.all_safe_keep, seed=s.seed)
    val_probe = probe_rule(all_safe_keep=1.0, seed=s.seed)
    population = [r for _, ep in train_eps for r in ep if is_decision(r) and consistent(r)]
    decision, w = training_weights(population, probe, s.all_safe_keep)
    weights = dict(zip((r["id"] for r in decision), w))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(s.seed)
    adapter.commit_state = any(sp.state == "commit" for sp in specs)
    if adapter.commit_state and not adapter.supports_commit_state:
        raise ValueError(f"{generator_id} has no commit pass: it has no prediction head")
    frozen_step0 = any(sp.frozen and sp.state != "commit" for sp in specs)
    if any(sp.frozen for sp in specs) and not s.frozen_states:
        raise ValueError("a head with `frozen` needs frozen_states")
    if s.frozen_states and not getattr(adapter, "supports_frozen_states", False):
        raise ValueError(f"{generator_id} cannot read the state after a held chunk")
    adapter.frozen_states = s.frozen_states
    adapter.frozen_step0 = frozen_step0
    if s.intent_head and not all("candidate_A_slide" in r for r in decision):
        raise ValueError("intent_head needs rows with candidate_A_slide")
    base = dataclasses.replace(adapter.field, intent_head=s.intent_head)
    loss_log = open(output / "losses.jsonl", "a")
    trainer = MultiHeadTrainer(specs, accumulate=s.accumulate, lr=s.lr, warmup=s.warmup, ema_decay=s.ema_decay, w=s.w, device=device,
                               seed=s.seed, base=base, commit_base=adapter.commit_field(base), loss_log=loss_log)
    trainer.rule = {"rule": s.rule, "tau": s.tau}
    run_config = {"generator": generator_id, "heads": [sp.name for sp in specs], "all_safe_keep": s.all_safe_keep, "seed": s.seed,
                  "max_rounds": s.max_rounds, "max_steps": s.max_steps, "limit_train": s.limit_train, "accumulate": s.accumulate,
                  "lr": s.lr, "warmup": s.warmup, "ema_decay": s.ema_decay, "w": s.w, "tau": s.tau, "rule": s.rule,
                  "intent_head": s.intent_head, "frozen_states": s.frozen_states, **adapter.run_config()}
    state_path, done, rounds_log, rng_state = output / "checkpoint.pt", [], [], None
    if state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state["config"] != run_config:
            raise ValueError(f"{state_path} was written by a different run: {state['config']} != {run_config}")
        trainer.load_state_dict(state["trainer"])
        done, rounds_log, rng_state = state["done"], state["rounds"], state["rng"]
        print(json.dumps({"resumed": state_path.name, "videos_done": len(done), "states": trainer.seen}), flush=True)
        state = None
    if s.max_steps is not None:
        for head in trainer.heads:
            head.max_steps = round(s.max_steps * s.accumulate / head.accumulate)
    began = time.perf_counter()
    adapter.load(adapter.args)
    print(json.dumps({"generator": generator_id, "load_seconds": round(time.perf_counter() - began, 1)}), flush=True)
    if s.keep_activations:
        for head in trainer.heads:
            head.model.recompute_blocks = False
    adapter.prepare_text([ep[0]["instruction"] for _, ep in train_eps + val_eps + val_extra])
    if rng_state is not None:
        torch.set_rng_state(rng_state["cpu"].cpu())
        torch.cuda.set_rng_state_all([st.cpu() for st in rng_state["cuda"]])

    def save():
        loss_log.flush()
        atomic_save({"trainer": trainer.state_dict(), "done": done, "rounds": rounds_log, "config": run_config,
                     "rng": {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}}, state_path)

    def end_round(entry):
        entry["always_keep"] = trainer.last_round_baseline
        rounds_log.append(entry)
        print(json.dumps(entry), flush=True)
        save()

    current = round_of[train_eps[0][0]] if train_eps else 0
    stopped = None
    for index, (pair_id, rows) in enumerate(train_eps):
        if pair_id in done:
            current = round_of[pair_id]
            continue
        if trainer.all_capped():
            stopped = "max_steps"
            break
        if round_of[pair_id] != current:
            if trainer.base_round["states"]:
                end_round({"round": current, "videos": len(done), "states": trainer.seen, "steps": trainer.steps, "heads": trainer.finish_round()})
            current = round_of[pair_id]
        began = time.perf_counter()
        adapter.announce(with_states(train_eps[index + 1:index + 1 + s.prefetch], probe, s.frozen_states))
        states, _ = adapter.replay(
            rows, lambda h, c, t, row, commit_hidden=None: trainer.observe(h, c, t, row, weight=weights[row["id"]], commit_hidden=commit_hidden),
            layers=trainer.all_layers, seed=episode_seed(pair_id, s.seed), probe=probe, **({"frozen": trainer.observe_frozen} if s.frozen_states else {}))
        if states != expected_states(rows, probe):
            raise RuntimeError(f"{pair_id}: {states} states reached the heads, expected {expected_states(rows, probe)}")
        done.append(pair_id)
        print(json.dumps({"video": pair_id, "round": current, "states": states, "steps": trainer.steps,
                          "seconds": round(time.perf_counter() - began, 1)}), flush=True)
        if s.checkpoint_every and len(done) % s.checkpoint_every == 0:
            save()
    if trainer.base_round["states"]:
        end_round({"round": current, "videos": len(done), "states": trainer.seen, "steps": trainer.steps, "heads": trainer.finish_round(),
                   "stop": stopped or "stream ended"})
    trainer.step()
    save()
    loss_log.close()
    adapter.forget_announced()
    final_eps = spread(val_eps, s.final_val_videos) + val_extra
    collector = Collector(trainer)
    frozen_collector = Collector(trainer, [h for h in trainer.heads if h.spec.state == "commit" or frozen_step0]) if s.frozen_states else None
    frozen_read = {"frozen": frozen_collector.observe_frozen} if frozen_collector is not None else {}
    for i, (pair_id, rows) in enumerate(final_eps):
        adapter.announce(with_states(final_eps[i + 1:i + 1 + s.prefetch], val_probe, s.frozen_states))
        adapter.replay(rows, collector.observe, layers=trainer.all_layers, seed=episode_seed(pair_id, s.seed), probe=val_probe, **frozen_read)
        print(json.dumps({"validation": i + 1, "of": len(final_eps), "video": pair_id}), flush=True)
    preds = collector.predictions()
    atomic_save(preds, output / "val_preds.pt")
    if frozen_collector is not None:
        atomic_save(frozen_collector.predictions(), output / "val_frozen_preds.pt")
    heads = {}
    for head in trainer.heads:
        heads[head.name] = {"spec": head.spec.to_dict(), "params": sum(p.numel() for p in head.model.parameters()),
                            "optimizer_steps": head.steps, "val": evaluate(preds[head.name], **trainer.rule)}
        atomic_save({"state_dict": head.ema.state_dict(), "config": head.model.config.to_dict(), "w": s.w, "sigma_a": SIGMA_A,
                     "head": head.spec.to_dict(), "prior": None, "generator": generator_id, "clearance_weighting": {"kind": "plain"},
                     "run_config": adapter.run_config()}, output / f"head_{head.name}.pt")
    report = {"generator": generator_id, "run_config": run_config, "train_videos": len(done), "train_states": trainer.seen,
              "train_frozen_states": trainer.frozen_seen, "optimizer_steps": {h.name: h.steps for h in trainer.heads},
              "rounds": rounds_log, "val_videos": len(final_eps), "heads": heads}
    write_json(output / "report.json", json_ready(report))
    return report


def frozen_heads(adapter, payloads, *, w, seed, intent_head, device):
    """A trainer whose EMA models are the released heads' weights; no optimizer step ever runs."""
    specs = [HeadSpec(layers=tuple(p["head"]["layers"]), state=p["head"].get("state", "step0"), accumulate=p["head"].get("accumulate", 0),
                      lr_scale=p["head"].get("lr_scale", 1), replica=p["head"].get("replica", 0)) for p in payloads]
    base = dataclasses.replace(adapter.field, intent_head=intent_head)
    trainer = MultiHeadTrainer(specs, w=w, device=device, seed=seed, base=base, commit_base=adapter.commit_field(base))
    for head, p in zip(trainer.heads, payloads):
        config = {**FieldConfig(layers=tuple(p["config"]["layers"])).to_dict(), **p["config"], "layers": tuple(p["config"]["layers"])}
        built = {**head.ema.config.to_dict(), "layers": tuple(head.ema.config.layers)}
        if built != config:
            raise ValueError(f"head {head.name}: the file's architecture {config} is not the one this generator builds {built}")
        head.ema.load_state_dict(p["state_dict"])
        head.model.load_state_dict(p["state_dict"])
    return trainer


def task1_head_report(pred, point, mode, *, commit):
    """One head's Task 1 report from its predictions on the decision states and its operating point. A commit-pass head has no state
    at a route's first replayed chunk; those rows are left out."""
    if commit:
        pred = some_states(pred, pred["has_state"]) | {"id": np.asarray(pred["id"])[pred["has_state"]]}
    reward, safe = pred["reward_target"].numpy().astype(float), pred["safe_target"].numpy() >= 0.5
    choice = choose(pred["logits"], pred["intent_fit"], pred["candidate_mask"], keep_index=0, **point)
    picked, p_keep = choice["choice"].numpy(), choice["P"][:, 0].numpy()
    rows_ = np.arange(len(picked))
    regret = reward[rows_, reward.argmax(1)] - reward[rows_, picked]
    return {"mode": mode, "point": point, "states": int(len(picked)),
            "always_keep": selection_metrics(np.zeros(len(picked), int), reward, safe.astype(float)),
            "per_state": [{"id": str(i_), "video": str(v), "chunk": int(c), "choice": NAMES[int(k)], "p_keep": round(float(p), 4),
                           "keep_safe": bool(ks), "regret": round(float(r), 4)}
                          for i_, v, c, k, p, ks, r in zip(pred["id"], pred["video"], pred["chunk"], picked, p_keep, safe[:, 0], regret)]}


def task1(adapter, payloads, modes, rows_path, output, *, seed=202, w=W_LATERAL, prefetch=1, threads=16):
    """Task 1: the released heads on the decision states of the routes' true history (teacher-forced), no training.
    Writes <output>/task1_preds.pt and <output>/task1_report.json (per head and state: the choice, P(KEEP), whether KEEP is safe, regret)."""
    torch.set_num_threads(threads)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    trainer = frozen_heads(adapter, payloads, w=w, seed=seed, intent_head=True, device=device)
    adapter.commit_state = trainer.commit_state
    adapter.frozen_states = adapter.frozen_step0 = False
    eps = episodes(read_jsonl(rows_path))
    probe = probe_rule(all_safe_keep=1.0, seed=seed)
    adapter.load(adapter.args)
    adapter.prepare_text([ep[0]["instruction"] for _, ep in eps])
    collector = Collector(trainer)
    for i, (pair_id, rows) in enumerate(eps):
        adapter.announce(with_states(eps[i + 1:i + 1 + prefetch], probe))
        states, _ = adapter.replay(rows, collector.observe, layers=trainer.all_layers, seed=episode_seed(pair_id, seed), probe=probe)
        print(json.dumps({"task1": i + 1, "of": len(eps), "video": pair_id, "states": states}), flush=True)
    preds = collector.predictions()
    atomic_save(preds, output / "task1_preds.pt")
    heads = {head.name: task1_head_report(preds[head.name], payload["selection"], mode, commit=head.spec.state == "commit")
             for head, payload, mode in zip(trainer.heads, payloads, modes)}
    report = {"generator": adapter.generator_id, "videos": len(eps), "heads": heads}
    write_json(output / "task1_report.json", json_ready(report))
    return report
