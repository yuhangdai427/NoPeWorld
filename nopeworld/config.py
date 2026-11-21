"""YAML configuration of world models and methods (configs/world_models/<id>.yaml, configs/methods/<id>.yaml)."""
from __future__ import annotations

import yaml

from .paths import REPO

CONFIGS = REPO / "configs"
METHODS = ("raw", "text_prompt", "external_depth", "nopeworld_collision", "nopeworld_prediction")
TASKS = ("task1", "task2", "task2_open", "task3")
MODES = ("collision", "prediction")


def read_yaml(path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def generator_config(generator_id: str) -> dict:
    path = CONFIGS / "world_models" / f"{generator_id}.yaml"
    if not path.is_file():
        raise ValueError(f"no world-model config {path.name}")
    config = read_yaml(path)
    if config.get("id") != generator_id:
        raise ValueError(f"{path.name} names generator {config.get('id')!r}")
    return config


def method_config(method_id: str) -> dict:
    if method_id not in METHODS:
        raise ValueError(f"unknown method {method_id!r}: one of {METHODS}")
    config = read_yaml(CONFIGS / "methods" / f"{method_id}.yaml")
    if config.get("id") != method_id:
        raise ValueError(f"{method_id}.yaml names method {config.get('id')!r}")
    return config


def generator_ids() -> list[str]:
    return sorted(p.stem for p in (CONFIGS / "world_models").glob("*.yaml"))


def modes_of(config: dict) -> list[str]:
    return [m for m in MODES if config.get("heads", {}).get(m)]


def method_available(generator: dict, method_id: str) -> bool:
    """Every generator has raw, text_prompt and external_depth; a head method needs that mode's head."""
    method = method_config(method_id)
    mode = method.get("head_mode")
    return mode is None or mode in modes_of(generator)


def as_args(values: dict) -> list[str]:
    """A config's generator flags as command-line arguments for the adapter's own parser."""
    argv = []
    for key, value in (values or {}).items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, (list, tuple)):
            argv += [flag, *map(str, value)]
        else:
            argv += [flag, str(value)]
    return argv
