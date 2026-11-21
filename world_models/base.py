"""What the trainer and the rollout ask of a world model. One adapter per world model; it runs in that model's environment.

Training:
    field                 the head's FieldConfig base (token grid, token width, camera layout)
    add_arguments(ap)     generator options of the replay
    load(args)            builds what the replay needs
    prepare_text(prompts) encodes all prompts of the run once
    announce(rows)        the videos that come next (prepared in the background)
    replay(rows, observe, *, layers, seed, probe) -> (states, seconds per chunk)
                          one teacher-forced replay; observe(hidden, camera, interval_ends, row) per probed decision state
    run_config()          the flags that change the hidden states; a head is only valid under the same flags

Generation:
    head_must_match(args), add_generation_arguments(ap), load_generation(args) -> pipe, prepare_generation_text(pipe, packages)
    generate_case(pipe, package, out, *, track, head, payload, ...) -> (log, video)
    case_meta(args, package, log)   what is written next to the video
"""
from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
from unittest import mock

import torch

from nopeworld.head.model import FieldConfig
from nopeworld.paths import generators_path, root


class HistoryDoesNotFit(ValueError):
    """The route's hand-off frame leaves no room for the history asked for."""


def gpu_cache_held(hold):
    """Suppresses torch.cuda.empty_cache inside generator code while the memory is to be held between chunks."""
    return mock.patch.object(torch.cuda, "empty_cache", lambda: None) if hold else contextlib.nullcontext()


KEEP_TOLERANCE = 1e-2


def check_keep(error, chunk, record):
    """Candidate 0 must be what the chunk is about to run, in the numbers the generator reads."""
    if error > KEEP_TOLERANCE:
        raise ValueError(f"chunk {chunk}: the head's KEEP differs from the commanded control by {error:.4f}")
    record["keep_rebuild_error"] = round(error, 6)


def prompt_sha256(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def relative_to_roots(path) -> str:
    """A path written into a run's records: relative to the generators or data root when it lies under one."""
    path = Path(path).absolute()
    for name in ("generators", "data", "weights"):
        base = root(name).absolute()
        try:
            return f"{name}/{path.relative_to(base)}"
        except ValueError:
            continue
    return path.name


class Adapter:
    generator_id: str = ""
    field: FieldConfig = FieldConfig()
    supports_commit_state = False     # the generator has a commit pass whose tokens a prediction head reads
    commit_state = False              # the trainer sets it when a prediction head is in the run
    supports_frozen_states = False    # the replay can also read the state after one held chunk
    frozen_states = False
    model_commit: str = ""            # the generator repository's commit this adapter is pinned to
    control_file: str = "poses.npy"   # the package file a route's command is compiled from
    weights: tuple = ()               # the generation arguments that name the generator's weights
    step0_t = None
    clock = None

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        paths = self.config.get("paths", {})
        self.paths = {name: generators_path(value) for name, value in paths.items()}

    def path(self, name) -> Path:
        if name not in self.paths:
            raise KeyError(f"configs/world_models/{self.generator_id}.yaml has no paths.{name}")
        return self.paths[name]

    def add_arguments(self, ap):
        pass

    def load(self, args):
        raise NotImplementedError

    def prepare_text(self, prompts):
        raise NotImplementedError

    def announce(self, rows_of_next_videos):
        pass

    def forget_announced(self):
        prefetcher = getattr(self, "prefetcher", None)
        if prefetcher is not None:
            prefetcher.cancel_all()

    def replay(self, rows, observe, *, layers, seed, probe=None):
        raise NotImplementedError

    def first_probe(self, chunk, t, hidden, **extra):
        if not getattr(self, "first_probe_printed", False):
            self.first_probe_printed = True
            print(json.dumps({"first_probe": {"chunk": int(chunk), "t": float(t), "hidden_shape": list(hidden.shape), "hidden_dtype": str(hidden.dtype), **extra}}), flush=True)

    def release_gpu(self):
        if torch.cuda.is_available() and not getattr(getattr(self, "args", None), "hold_gpu_memory", False):
            torch.cuda.empty_cache()

    def held_gpu(self):
        return gpu_cache_held(getattr(getattr(self, "args", None), "hold_gpu_memory", False))

    def prepared(self, inputs):
        if self.clock is not None:
            self.clock.mark("prepared_frames_and_latents")
            self.clock.prepare_seconds = inputs.seconds

    def mark(self, part):
        if self.clock is not None:
            self.clock.mark(part)

    def commit_field(self, field):
        return field

    def run_config(self) -> dict:
        return {}

    def run_notes(self) -> dict:
        return {}

    def head_must_match(self, args) -> dict:
        return {}

    def head_flag_defaults(self) -> dict:
        return {}

    def add_generation_arguments(self, ap):
        pass

    def load_generation(self, args):
        raise NotImplementedError

    def prepare_generation_text(self, pipe, packages):
        pass

    def generate_case(self, pipe, package, out, *, track, head=None, payload=None, observe_only=False, force=None,
                      scorer=False, history_chunks=0):
        raise NotImplementedError

    def case_meta(self, args, package, log) -> dict:
        """<route>.meta.json: the video (fps, frames, size, crop), what it was asked (seed, text sha256, control file), what made it."""
        control = Path(package) / self.control_file
        return {"case": log["case"], "generator": self.generator_id, "model_commit": self.model_commit,
                "weights": {name: relative_to_roots(getattr(args, name)) for name in self.weights},
                "track": log["track"], "policy": log.get("policy"), "history_chunks": log["history_chunks"], "retime": log.get("retime", False),
                "fps": log["fps"], "frames": log["frames"], "size": log["size"], "crop": log["crop"], "seed": log["seed"],
                "prompt_sha256": log["prompt_sha256"], "control_file": control.name, "control_sha256": hashlib.sha256(control.read_bytes()).hexdigest(),
                "decisions_file": f"{log['case']}.decisions.json"}
