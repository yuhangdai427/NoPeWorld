"""Text-prompt baseline: each route's own prompt, a space, then a sentence naming the obstacle ahead; nothing else changes."""
from __future__ import annotations

import json
import os
from pathlib import Path

REWRITTEN = ("prompt.txt", "prompts.json")


def appended(text, added):
    return text.strip() + " " + added.strip()


def overlay_package(package, out, entry):
    """out: a folder with links to every file of `package` except the prompt files, which are written with `entry['added']` appended."""
    package = Path(package).resolve()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for f in package.iterdir():
        target = out / f.name
        if f.name in REWRITTEN:
            continue
        if target.is_symlink() or target.exists():
            if target.resolve() == f.resolve():
                continue
            raise FileExistsError(f"{target} exists and is not {f}")
        os.symlink(f, target)
    own = (package / "prompt.txt").read_text().strip()
    if not entry or not entry.get("added"):
        (out / "prompt.txt").write_text(own + "\n")
        if (package / "prompts.json").exists():
            (out / "prompts.json").write_text((package / "prompts.json").read_text())
        return False
    added = entry["added"]
    (out / "prompt.txt").write_text(appended(own, added) + "\n")
    if (package / "prompts.json").exists():
        p = json.loads((package / "prompts.json").read_text())
        p["prompt"] = appended(p["prompt"], added)
        for generator, text in p.get("per_model", {}).items():
            p["per_model"][generator] = appended(text, added)
        p["case_prompt"] = dict(rule="own prompt + ' ' + added", added=added.strip())
        (out / "prompts.json").write_text(json.dumps(p, indent=1, ensure_ascii=False))
    return True


def text_table(task, group=None):
    """The added sentences of a task, route or case id -> entry."""
    from ...paths import data_path
    name = {"task2": "task2.json", "task2_open": "task2.json"}.get(task) or ("task3_dynamic.json" if group and group.startswith("dynamic") else "task3_static.json")
    return json.loads(data_path("text_prompts", name).read_text())
