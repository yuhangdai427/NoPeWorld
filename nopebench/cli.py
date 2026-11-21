"""Console scripts: nopebench-eval (Task 1, 2, 3 on a run folder) and nopebench-download (dataset and heads from the Hugging Face Hub)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from nopeworld.paths import set_roots

GENERATORS = ("lingbot_world_2", "hy_worldplay", "echo_wm", "evoke_turbo", "worldplay2", "hy_nomas")
METHODS = ("raw", "text_prompt", "external_depth", "nopeworld_collision", "nopeworld_prediction")


def run_info(run):
    """The run folder's own description (run.json: generator, method), when there is one."""
    f = Path(run) / "run.json"
    return json.loads(f.read_text()) if f.is_file() else {}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=float) + "\n")
    tmp.replace(path)


def eval_task1(a):
    from . import task1
    row = task1.evaluate_run(a.runs, a.generator)
    write_json(a.out or Path(a.runs[0]) / "task1_table.json", row)
    print(json.dumps(row, indent=1))


def eval_task2(a):
    from . import data
    from .task2 import evaluate, metrics
    stages = a.stages
    runs = [("task2", a.run)] + ([("task2_open", a.open_run)] if a.open_run else [])
    if a.raw_run and a.raw_run != a.run:
        runs.append(("task2", a.raw_run))
    for task, run in runs:
        ids = a.routes or data.route_ids(task)
        work = [s for s in stages if s in ("convert", "measure", "judge")]
        if work:
            evaluate.evaluate_run(run, {r: data.route_dir(r, task) for r in ids}, data.scenes_root(), stages=work, backend=a.recon)
    if "metrics" not in stages:
        return
    if a.recon != "vggt_omega":
        print("note: the paper's Task 2 numbers are on the VGGT-Omega reconstruction", file=sys.stderr)
    routes = data.task2_routes("task2", a.routes)
    eval_dir = lambda run: Path(run) / evaluate.eval_dir_name(a.recon)
    videos = metrics.read_videos(eval_dir(a.run), routes, a.recon)
    raw = videos if (a.method == "raw" and not a.raw_run) else (metrics.read_videos(eval_dir(a.raw_run), routes, a.recon) if a.raw_run else None)
    if raw is None:
        print("no --raw-run: SIS needs the same generator's raw run for its speed; SIS is left out", file=sys.stderr)
    open_routes = data.task2_routes("task2_open") if a.open_run else None
    open_videos = metrics.read_videos(eval_dir(a.open_run), open_routes, a.recon) if a.open_run else None
    table, per_route = metrics.task2_table(routes, videos, generator=a.generator, raw_videos=raw, open_routes=open_routes, open_videos=open_videos)
    table.update(method=a.method, recon=a.recon, rule="nopebench")
    write_json(a.out or eval_dir(a.run) / "table.json", table)
    write_json(eval_dir(a.run) / "per_route.json", per_route)
    show = {k: (v.get("value") if isinstance(v, dict) else v) for k, v in table.items() if k in ("sis", "violation", "back", "sfp", "slide", "success")}
    print(json.dumps(dict(generator=a.generator, method=a.method, incomplete=table["incomplete"], **show,
                          open_sfp=(table.get("open") or {}).get("sfp", {}).get("value"))))


def eval_task3(a):
    from . import data
    from .task3 import prompt, results
    out = Path(a.run) / "task3_eval"
    manifest_file = out / "manifest.json"
    if a.stage == "prepare":
        items, key = [], {}
        for group in a.groups or data.TASK3_GROUPS:
            for case_id in data.task3_case_ids(group):
                video = Path(a.run) / f"{case_id}.mp4"
                if not video.is_file():
                    continue
                meta = json.loads((data.task3_case(group, case_id)["full"] / "meta.json").read_text())
                n = prompt.extract_frames(video, out / "frames" / case_id)
                items.append(dict(id=case_id, **prompt.hint(meta, n, group)))
                key[case_id] = dict(group=group, case=case_id, generator=a.generator, method=a.method)
        write_json(manifest_file, dict(prompt_sha256=prompt.prompt_sha(), items=items, key=key))
        print(f"{len(items)} videos prepared in {out}")
        return
    manifest = json.loads(manifest_file.read_text())
    if not manifest["prompt_sha256"].startswith(prompt.PROMPT_SHA256):
        raise SystemExit(f"{manifest_file}: made with another prompt")
    items = {x["id"]: x for x in manifest["items"]}
    if a.stage == "judge":
        answers = out / f"answers_{a.judge}.jsonl"
        done = set(results.load_answers([answers], items)) if answers.is_file() else set()
        key = Path(a.api_key_file).expanduser().read_text().strip() if a.api_key_file else None
        for item_id, item in items.items():
            if item_id in done:
                continue
            note, rec = None, None
            for attempt in range(a.retries + 1):
                content, reasoning = prompt.ask(a.judge_url, a.judge_model, prompt.messages(item, out / "frames" / item_id, note),
                                                thinking=a.thinking, max_tokens=a.max_tokens, key=key)
                ans, why = prompt.validate(item, prompt.parse_json(content))
                rec = dict(id=item_id, valid=ans is not None, answer=ans, why=why, raw=content, reasoning=reasoning, attempt=attempt,
                           prompt_sha256=prompt.prompt_sha())
                if ans is not None:
                    break
                note = why
            with answers.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        return
    by_judge = {}
    for spec in a.answers:
        judge, _, path = spec.partition("=")
        answers = results.load_answers([path], items)
        by_judge[judge] = results.cells({i: results.coarse(x) for i, x in answers.items()}, manifest["key"])
    res = dict(cells=by_judge)
    if all(j in by_judge for j in results.MAIN_JUDGES):
        res["violation"] = results.main_violation(by_judge, a.generator, a.method)
    write_json(a.out or out / "results.json", res)
    print(json.dumps(res.get("violation") or {j: len(c) for j, c in by_judge.items()}))


def eval_main(argv=None):
    ap = argparse.ArgumentParser(prog="nopebench-eval", description="Evaluate a run folder on Task 1, 2 or 3 of NoPeBench.")
    ap.add_argument("--task", choices=("task1", "task2", "task3"), required=True)
    ap.add_argument("--run", help="the run folder (nopeworld-infer's output)")
    ap.add_argument("--runs", nargs="+", help="Task 1: run folders whose heads are scored together (one per mode, or one --method nopeworld run)")
    ap.add_argument("--generator", choices=GENERATORS, help="default: run.json of the run folder")
    ap.add_argument("--method", choices=METHODS, help="default: run.json of the run folder")
    ap.add_argument("--out", help="the table file (default: inside the run folder)")
    ap.add_argument("--data-root")
    ap.add_argument("--weights-root")
    ap.add_argument("--generators-root")
    t2 = ap.add_argument_group("task2")
    t2.add_argument("--stages", nargs="+", default=["convert", "measure", "judge", "metrics"], choices=("convert", "measure", "judge", "metrics"),
                    help="measure needs the GPU (VGGT-Omega, SAM 2); the others run on the CPU")
    t2.add_argument("--raw-run", help="the same generator's raw run (SIS reads its speed); not needed for the raw method itself")
    t2.add_argument("--open-run", help="the run on the 50 open routes (task2_open): the Open column")
    t2.add_argument("--recon", choices=("vggt_omega", "da3"), default="vggt_omega")
    t2.add_argument("--routes", nargs="*", help="only these routes")
    t3 = ap.add_argument_group("task3")
    t3.add_argument("--stage", choices=("prepare", "judge", "score"), default="prepare")
    t3.add_argument("--groups", nargs="*", help="Task 3 groups (default: all four)")
    t3.add_argument("--judge", default="judge", help="the judge's name (answers_<judge>.jsonl)")
    t3.add_argument("--judge-url", default="http://127.0.0.1:8000", help="an OpenAI-compatible chat server")
    t3.add_argument("--judge-model", default="")
    t3.add_argument("--api-key-file", default="")
    t3.add_argument("--thinking", action="store_true")
    t3.add_argument("--max-tokens", type=int, default=4096)
    t3.add_argument("--retries", type=int, default=2)
    t3.add_argument("--answers", nargs="*", default=[], help="score: <judge>=<answers.jsonl> per judge (the main table: claude and gpt6)")
    a = ap.parse_args(argv)
    set_roots(a.data_root, a.weights_root, a.generators_root)
    if (a.run is None) == (a.runs is None):
        ap.error("give --run, or for Task 1 --run or --runs")
    if a.runs and a.task != "task1":
        ap.error("--runs is for Task 1")
    a.runs = a.runs or [a.run]
    a.run = a.runs[0]
    info = run_info(a.run)
    a.generator = a.generator or info.get("generator")
    a.method = a.method or info.get("method")
    if a.task != "task1" and a.method is None:
        ap.error("--method is needed (no run.json in the run folder)")
    if a.generator is None:
        ap.error("--generator is needed (no run.json in the run folder)")
    {"task1": eval_task1, "task2": eval_task2, "task3": eval_task3}[a.task](a)


def download_main(argv=None):
    ap = argparse.ArgumentParser(prog="nopebench-download", description="Download NoPeBench (tar archives, checked and extracted) and the NoPeWorld heads from the Hugging Face Hub.")
    ap.add_argument("--what", choices=("data", "weights", "all"), required=True)
    ap.add_argument("--tasks", nargs="*", choices=("task1", "task2", "task2_open", "task3"), help="only the archives these tasks need")
    ap.add_argument("--split", nargs="*", choices=("train", "val"), help="only the archives of these splits")
    ap.add_argument("--cases", nargs="*", help="only the archives holding these route, case or episode ids (plus the scene maps, and the rows for --split train)")
    ap.add_argument("--keep-archives", action="store_true", help="keep each tar file after it is extracted")
    ap.add_argument("--generators", nargs="*", choices=GENERATORS, help="only these world models' heads")
    ap.add_argument("--data-root")
    ap.add_argument("--weights-root")
    a = ap.parse_args(argv)
    set_roots(a.data_root, a.weights_root)
    from . import download
    if a.what in ("data", "all"):
        parts = list(a.tasks or []) + list(a.split or [])
        print(json.dumps(download.download_data(parts or None, cases=a.cases, keep_archives=a.keep_archives), indent=1))
    if a.what in ("weights", "all"):
        print(download.download_weights(a.generators))


if __name__ == "__main__":
    eval_main()
