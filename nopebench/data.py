"""Readers of the NoPeBench dataset layout (the dataset's LAYOUT.md); paths are relative to the data root (NOPE_DATA)."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from nopeworld.paths import data_path, weights_path


MANIFEST = "manifest"
SCENES = "scenes"
TASK1_ROWS = "task1/rows"                 # task1/rows/<generator id>.jsonl.gz
TASK2_ROUTES = "task2/routes"             # task2/routes/<route_id>/
TASK2_OPEN_ROUTES = "task2/open_routes"
TASK3 = "task3"                           # task3/<group>/<case_id>/
TEXT_PROMPTS = "text_prompts"
ROUTE_LISTS = {"task1": "task1_routes.txt", "task2": "task2_routes.txt", "task2_open": "task2_open_routes.txt"}
TASK3_GROUPS = ("static_first_person", "static_third_person", "dynamic_first_person", "dynamic_third_person")
ROUTE_JSON, CASE_DIR, REFERENCE_DIR = "route.json", "case", "reference"
PACKAGE = {16: "fps16", 24: "fps24"}
TASK3_CASE_JSON, TASK3_FULL, TASK3_DEFAULT = "case.json", "lingbot_world_2", "default"
TEXT_PROMPT_FILES = {"task2": "task2.json", "task3_static": "task3_static.json", "task3_dynamic": "task3_dynamic.json"}
HEADS = "heads"                           # heads/<generator id>/<mode>/head.pt (+ config.json) in the model repo

def read_lines(path):
    """The ids of a list file, one per line."""
    return [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]


def read_jsonl(path):
    """The rows of a .jsonl or .jsonl.gz file."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def route_ids(task):
    """The route ids of task1, task2 or task2_open (manifest/<task>_routes.txt)."""
    return read_lines(data_path(MANIFEST, ROUTE_LISTS[task]))


def route_dir(route_id, task="task2"):
    return data_path(TASK2_OPEN_ROUTES if task == "task2_open" else TASK2_ROUTES, route_id)


def route_package(route_id, fps=16, task="task2"):
    """The generator input of a route at its native clock (the fps16 or fps24 folder)."""
    return route_dir(route_id, task) / PACKAGE[int(fps)]


def route_info(route_id, task="task2"):
    """route.json of a route: route_id, scene, version, lead_s, reference_starts_at, case_kind."""
    return json.loads((route_dir(route_id, task) / ROUTE_JSON).read_text())


def scenes_root():
    return data_path(SCENES)


def task2_routes(task="task2", ids=None):
    """The routes as the Task 2 metrics read them (nopebench.task2.metrics.load_route on each fps16 package)."""
    from .task2.metrics import load_route
    out = []
    for route_id in ids or route_ids(task):
        info = route_info(route_id, task)
        out.append(load_route(route_package(route_id, 16, task), route_id=route_id, scene=info.get("scene"), version=info.get("version")))
    return out


def task1_rows(generator):
    return read_jsonl(data_path(TASK1_ROWS, f"{generator}.jsonl.gz"))


def task3_case_ids(group):
    return read_lines(data_path(MANIFEST, f"task3_{group}.txt"))


def task3_case(group, case_id):
    """A Task 3 case: its case.json, the full 8 s input package and the package the other generators read."""
    d = data_path(TASK3, group, case_id)
    return dict(case=json.loads((d / TASK3_CASE_JSON).read_text()), full=d / TASK3_FULL, default=d / TASK3_DEFAULT)


def text_prompts(name):
    """The text-prompt baseline sentences: task2, task3_static or task3_dynamic."""
    return json.loads(data_path(TEXT_PROMPTS, TEXT_PROMPT_FILES[name]).read_text())


def head_path(generator, mode):
    return weights_path(HEADS, generator, mode, "head.pt")
