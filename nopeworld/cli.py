"""Command-line entry points: nopeworld-train and nopeworld-infer."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from .config import METHODS, MODES, TASKS, as_args, generator_config, generator_ids, method_available, method_config, modes_of
from .paths import data_path, set_roots

IGNORED_TRAINING_KEYS = ("skip_unused_probes", "save_val_preds", "hold_gpu_memory")
BOTH_HEADS = "nopeworld"


def add_roots(ap):
    ap.add_argument("--data-root", help="dataset root (default $NOPE_DATA or ./data)")
    ap.add_argument("--weights-root", help="head weights root (default $NOPE_WEIGHTS or ./weights)")
    ap.add_argument("--generators-root", help="generator checkpoints and code (default $NOPE_GENERATORS or ./generators)")


def training_namespace(adapter, generator, settings_cls, extra=()):
    """Splits the config's `training:` block into trainer settings and the adapter's own replay flags."""
    fields = {f.name for f in dataclasses.fields(settings_cls)}
    block = dict(generator.get("training") or {})
    trainer = {k: v for k, v in block.items() if k in fields}
    flags = {k: v for k, v in block.items() if k not in fields and k not in IGNORED_TRAINING_KEYS}
    ap = argparse.ArgumentParser(prog=f"{generator['id']} replay")
    adapter.add_arguments(ap)
    args = ap.parse_args(as_args(flags) + list(extra))
    return trainer, args


def train_main(argv=None):
    from world_models import get_adapter
    from .training.selection import select_head
    from .training.trainer import TrainSettings, spec_of, train
    ap = argparse.ArgumentParser(prog="nopeworld-train", description="Train a NoPeWorld head on a frozen generator's replay of the training videos, then fix its operating point on validation.")
    ap.add_argument("--generator", required=True, choices=generator_ids())
    ap.add_argument("--mode", required=True, choices=MODES + ("both",))
    ap.add_argument("--output", required=True, type=Path, help="run folder (checkpoint, losses.jsonl, val_preds.pt, report.json, heads/)")
    ap.add_argument("--max-steps", type=int, help="optimizer steps (default: the config's)")
    ap.add_argument("--limit-train", type=int, help="only the first this many videos of the stream")
    ap.add_argument("--limit-val", type=int, help="only this many validation videos")
    ap.add_argument("--no-validation-extra", action="store_true", help="leave the near-wall validation clips out")
    ap.add_argument("--no-select", action="store_true", help="train only; do not fix the operating point")
    ap.add_argument("--prefetch", type=int, help="videos prepared ahead in a background thread (default: the config's; 0 = none)")
    add_roots(ap)
    a, extra = ap.parse_known_args(argv)
    set_roots(a.data_root, a.weights_root, a.generators_root)
    generator = generator_config(a.generator)
    modes = modes_of(generator) if a.mode == "both" else [a.mode]
    if not all(m in modes_of(generator) for m in modes):
        ap.error(f"{a.generator} has no {a.mode} head")
    adapter = get_adapter(a.generator, generator)
    trainer_flags, args = training_namespace(adapter, generator, TrainSettings, extra)
    rows = data_path("train", "rows", a.generator)
    val = data_path("val", "rows", a.generator)
    settings = TrainSettings(rows_train=rows / "rows.jsonl.gz", stream=rows / "stream.json", rows_val=val / "rows.jsonl.gz",
                             rows_val_extra=None if a.no_validation_extra else val / "rows_near_wall.jsonl.gz", output=a.output, **trainer_flags)
    if a.max_steps is not None:
        settings.max_steps = a.max_steps
    settings.limit_train, settings.limit_val = a.limit_train or settings.limit_train, a.limit_val or settings.limit_val
    if a.prefetch is not None:
        settings.prefetch = a.prefetch
    specs = [spec_of(generator["heads"][m]) for m in modes]
    settings.frozen_states = settings.frozen_states or any(s.frozen for s in specs)
    for key in ("prefetch", "seed", "threads"):
        setattr(args, key, getattr(settings, key))
    args.hold_gpu_memory = True
    adapter.args = args
    report = train(settings, adapter, specs, a.generator)
    if not a.no_select:
        for mode, spec in zip(modes, specs):
            summary = select_head(a.output, spec.name, state=spec.state, limit=generator["heads"][mode]["missed_limit"],
                                  generator=a.generator, mode=mode, out=a.output / "heads" / a.generator / mode / "head.pt")
            print(json.dumps(summary), flush=True)
    print(json.dumps({k: report[k] for k in ("train_videos", "train_states", "optimizer_steps", "val_videos")}), flush=True)


def infer_main(argv=None):
    from .inference.rollout import TASK3_GROUPS, run
    ap = argparse.ArgumentParser(prog="nopeworld-infer", description="Generate with a generator and a method: Task 2 routes, open routes, Task 3 cases; or Task 1 decisions on the true history.")
    ap.add_argument("--generator", required=True, choices=generator_ids())
    ap.add_argument("--method", required=True, choices=METHODS + (BOTH_HEADS,),
                    help=f"{BOTH_HEADS} (Task 1 only): all of the world model's heads in one replay, as in the paper's Task 1 table")
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--group", choices=TASK3_GROUPS, help="Task 3 group")
    ap.add_argument("--cases", nargs="*", help="route or case ids (default: the task's whole list)")
    ap.add_argument("--case-file", type=Path, help="ids, one per line")
    ap.add_argument("--seeds", type=Path, help="JSON {route id: seed}; other routes use sha256(route id)")
    ap.add_argument("--out", required=True, type=Path)
    add_roots(ap)
    a, extra = ap.parse_known_args(argv)
    set_roots(a.data_root, a.weights_root, a.generators_root)
    generator = generator_config(a.generator)
    if a.method == BOTH_HEADS and a.task != "task1":
        ap.error(f"--method {BOTH_HEADS} is for Task 1; generation takes one method")
    if a.method != BOTH_HEADS and not method_available(generator, a.method):
        ap.error(f"{a.generator} has no {a.method}")
    cases = list(a.cases or []) + (a.case_file.read_text().split() if a.case_file else [])
    if a.task == "task1":
        task1_main(a, generator, cases, extra)
        return
    seeds = json.loads(a.seeds.read_text()) if a.seeds else None
    run(a.generator, a.method, a.task, a.out, routes=cases or None, group=a.group, seeds=seeds, extra_args=extra)


def task1_main(a, generator, cases, extra):
    """Task 1: the heads on every decision state of the routes' true history. `--method nopeworld` scores all of the generator's heads
    in one replay (the paper's Table 3); a head method scores that mode's head alone."""
    import torch
    from world_models import get_adapter
    from .head.model import read_head_file
    from .inference.methods.nopeworld import head_path
    from .training.trainer import TrainSettings, task1
    modes = modes_of(generator) if a.method == BOTH_HEADS else [method_config(a.method).get("head_mode")]
    if None in modes:
        raise SystemExit(f"task1 scores heads: --method {BOTH_HEADS}, nopeworld_collision or nopeworld_prediction")
    adapter = get_adapter(a.generator, generator)
    _, args = training_namespace(adapter, generator, TrainSettings, extra)
    args.prefetch, args.seed, args.threads, args.hold_gpu_memory = int((generator.get("training") or {}).get("prefetch", 1)), 202, 16, True
    adapter.args = args
    payloads = [read_head_file(head_path(a.generator, mode)) for mode in modes]
    rows = data_path("task1", "rows", f"{a.generator}.jsonl.gz")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"generator": a.generator, "method": a.method, "task": "task1", "modes": modes,
                                              "cases": cases or None}, indent=1) + "\n")
    if cases:
        from .training.replay import read_jsonl
        import gzip
        wanted = set(cases)
        subset = out / "task1_rows.jsonl.gz"
        with gzip.open(subset, "wt") as fh:
            for row in read_jsonl(rows):
                if row["pair_id"].split("__lead")[0] in wanted:
                    fh.write(json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in row.items()}) + "\n")
        rows = subset
    task1(adapter, payloads, modes, rows, out, prefetch=args.prefetch)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    infer_main()
