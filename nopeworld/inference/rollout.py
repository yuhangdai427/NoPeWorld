"""Closed-loop generation of one generator with one method on Task 2, its open routes, or Task 3.

Per route the generator starts from the route's true history (Task 2) or start image (Task 3) and generates chunk by chunk from the
commanded motion. raw and text_prompt run it as it is; external_depth and the two NoPeWorld modes decide every chunk: the decision
replaces the command when it is unsafe and the chunk is generated with the chosen candidate. Next to each <route>.mp4 go
<route>.decisions.json (every chunk's candidates, P and choice) and <route>.meta.json (seed, frames, crop, fps), the latter last:
a route with both files is done and a rerun skips it.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from world_models import get_adapter
from world_models.base import HistoryDoesNotFit

from ..config import as_args, generator_config, method_available, method_config
from ..paths import data_path
from .methods.nopeworld import load_released_head
from .methods.text_prompt import overlay_package, text_table

TASK3_GROUPS = ("static_first_person", "static_third_person", "dynamic_first_person", "dynamic_third_person")
KEEP_GOING_LIMIT = 5


def route_list(task, group=None):
    name = {"task2": "task2_routes.txt", "task2_open": "task2_open_routes.txt"}.get(task) or f"task3_{group}.txt"
    return data_path("manifest", name).read_text().split()


def package_of(generator, task, route, group=None):
    """The folder a generator reads for one route."""
    if task == "task2":
        return data_path("task2", "routes", route, generator["packages"])
    if task == "task2_open":
        return data_path("task2", "open_routes", route, generator["packages"])
    view = "lingbot_world_2" if generator["id"] == "lingbot_world_2" else "default"
    return data_path("task3", group, route, view)


def depth_package_of(task, group=None):
    """The 16 fps package the depth process reads for a route: Task 2 fps16/, Task 3 the case's own package."""
    def find(route, generator=None):
        if task in ("task2", "task2_open"):
            return data_path("task2", "routes" if task == "task2" else "open_routes", route, "fps16")
        return data_path("task3", group, route, "default")
    return find


def generation_args(adapter, generator, method_id, extra=(), task="task2"):
    """The adapter's generation namespace: its parser on the config's flags, then the task's and the method's own where a model has them."""
    flags = dict(generator.get("generation") or {})
    flags.update((generator.get("generation_by_task") or {}).get(task, {}))
    flags.update((generator.get("generation_by_method") or {}).get(method_id, {}))
    ap = argparse.ArgumentParser(prog=f"{generator['id']} generation")
    adapter.add_generation_arguments(ap)
    args = ap.parse_args(as_args(flags) + list(extra))
    args.hold_gpu_memory = True
    args.async_score = False
    return args


class Method:
    """What one method asks of the adapter: the track, the head and its payload, the depth process, the package it reads."""

    def __init__(self, generator, method_id, task, group, out):
        self.id, self.config = method_id, method_config(method_id)
        self.generator, self.task, self.group, self.out = generator, task, group, Path(out)
        self.track = "A" if method_id in ("raw", "text_prompt") else "B"
        self.mode = self.config.get("head_mode")
        self.table = text_table(task, group) if method_id == "text_prompt" else None
        self.head = self.payload = self.depth = None

    def start_depth(self):
        """The depth process starts before the generator loads: one that cannot start costs no generator load."""
        if self.id == "external_depth":
            from .methods.external_depth import Da3Stop, DepthSpec
            selection = (self.generator.get("external_depth") or {}).get("selection")
            if selection is None:
                from ..head.model import read_head_file
                from .methods.nopeworld import head_path
                selection = read_head_file(head_path(self.generator["id"], "collision"))["selection"]
            self.payload = {"selection": dict(selection)}
            offload = bool((self.generator.get("external_depth") or {}).get("offload", False))
            self.depth = DepthSpec(Da3Stop(self.out / "depth_process.log", offload=offload), depth_package_of(self.task, self.group))

    def load(self, adapter, args, device):
        if self.mode:
            self.head, self.payload = load_released_head(adapter, args, self.mode, device)

    def package(self, route, package):
        if self.table is None:
            return package
        target = self.out / "_text_prompt_packages" / route / package.name
        overlay_package(package, target, self.table.get(route))
        return target

    def close(self):
        if self.depth is not None:
            self.depth.da3.close()


def run_route(adapter, pipe, args, method, route, package, out, history_chunks, seed=None):
    info = None
    for chunks in range(history_chunks, -1, -1):
        try:
            info, _ = adapter.generate_case(pipe, package, out, track=method.track, head=method.head, payload=method.payload,
                                            scorer=False, history_chunks=chunks,
                                            **({} if method.depth is None else {"depth": method.depth}), **({} if seed is None else {"seed": seed}))
            break
        except HistoryDoesNotFit:
            if chunks == 0:
                raise
    info = {k: v for k, v in info.items() if k not in ("chunks", "executed_actions", "executed_controls")}
    info["policy"] = f"heads/{adapter.generator_id}/{method.mode}/head.pt" if method.mode else None
    info["method"] = method.id
    if seed is not None:
        info["seed"] = seed
    meta = adapter.case_meta(args, package, info)
    meta["method"] = method.id
    with open(out / "generation.jsonl", "a") as f:
        f.write(json.dumps(info) + "\n")
    print(json.dumps(info), flush=True)
    tmp = out / f"{route}.meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.replace(out / f"{route}.meta.json")


def run(generator_id, method_id, task, out, *, routes=None, group=None, seeds=None, keep_going=True, extra_args=()):
    """Generates every route of the list that is not done yet into `out`."""
    generator = generator_config(generator_id)
    if not method_available(generator, method_id):
        raise ValueError(f"{generator_id} has no {method_id}: its generator keeps no memory forward, so it has no prediction head")
    if task == "task3" and group not in TASK3_GROUPS:
        raise ValueError(f"task3 needs a group: one of {TASK3_GROUPS}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    routes = list(dict.fromkeys(routes or route_list(task, group)))
    seed_table = {k: int(v) for k, v in (seeds or {}).items()}
    done = lambda r: (out / f"{r}.mp4").exists() and (out / f"{r}.meta.json").exists()
    todo = [r for r in routes if not done(r)]
    (out / "run.json").write_text(json.dumps({"generator": generator_id, "method": method_id, "task": task, "group": group,
                                              "routes": len(routes)}, indent=1))
    print(json.dumps({"generator": generator_id, "method": method_id, "task": task, "group": group, "routes": len(routes), "todo": len(todo)}), flush=True)
    if not todo:
        return
    adapter = get_adapter(generator_id, generator)
    args = generation_args(adapter, generator, method_id, extra_args, task)
    adapter.args = args
    method = Method(generator, method_id, task, group, out)
    began = time.perf_counter()
    method.start_depth()
    pipe = adapter.load_generation(args)
    method.load(adapter, args, pipe.device if hasattr(pipe, "device") else torch.device("cuda"))
    history = generator.get("history_chunks", 0) if task in ("task2", "task2_open") else 0
    packages = {r: method.package(r, package_of(generator, task, r, group)) for r in todo}
    adapter.prepare_generation_text(pipe, [packages[r] for r in todo])
    print(json.dumps({"load_seconds": round(time.perf_counter() - began, 1), "selection": (method.payload or {}).get("selection")}), flush=True)
    failures = 0
    try:
        for second_pass in (False, True) if keep_going else (False,):
            for route in [r for r in todo if not done(r)]:
                try:
                    run_route(adapter, pipe, args, method, route, packages[route], out, history, seed_table.get(route))
                    failures = 0
                except Exception as exc:
                    if not keep_going:
                        raise
                    failures += 1
                    line = json.dumps({"case": route, "error": repr(exc), "second_pass": second_pass})
                    with open(out / "generation.jsonl", "a") as f:
                        f.write(line + "\n")
                    print(line, flush=True)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if failures >= KEEP_GOING_LIMIT:
                        raise SystemExit(f"{failures} failures in a row, the last: {exc!r}")
    finally:
        method.close()
    missing = [r for r in todo if not done(r)]
    if missing:
        print(json.dumps({"routes_missing": missing}), flush=True)
