"""Compares two run folders route by route: decoded frames, every chunk's decision record, and the run record.

    python tests/gpu/compare_runs.py --reference <dir> --candidate <dir> --routes <id> [<id> ...] [--out result.json]

A chunk record is compared on the keys both sides have, except wall-clock times and file locations; keys only one side has are listed.
Exit code 0 only if every route is identical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

VOLATILE = {"seconds", "policy", "depth_package", "control_file", "weights", "decisions_file", "model", "generator", "method",
            "seed_from", "at", "peak_gib", "reserved_gib", "load_seconds", "depth_log", "observe_only", "force", "force_from", "model_commit"}


def frames(path):
    import imageio_ffmpeg
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    meta = next(reader)
    w, h = meta["size"]
    out = [np.frombuffer(d, np.uint8).reshape(h, w, 3) for d in reader]
    return np.stack(out)


def timing(key):
    return key == "seconds" or key.endswith("_seconds") or key.startswith("seconds_")


def strip(record):
    """The record without wall-clock times (at any depth) and file locations."""
    if isinstance(record, dict):
        return {k: strip(v) for k, v in record.items() if k not in VOLATILE and not timing(k)}
    if isinstance(record, list):
        return [strip(v) for v in record]
    return record


def compare_records(a, b, where, problems, extra):
    a, b = strip(a), strip(b)
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            extra.add(f"{where}.{key}:{'reference' if key in a else 'candidate'} only")
        elif a[key] != b[key]:
            problems.append({"where": f"{where}.{key}", "reference": a[key], "candidate": b[key]})


def compare_route(ref, candidate, route):
    result = {"route": route}
    fa, fb = frames(ref / f"{route}.mp4"), frames(candidate / f"{route}.mp4")
    result["frames"] = [list(fa.shape), list(fb.shape)]
    result["frames_equal"] = bool(fa.shape == fb.shape and np.array_equal(fa, fb))
    if fa.shape == fb.shape:
        result["mean_abs_diff"] = float(np.abs(fa.astype(int) - fb.astype(int)).mean())
    problems, extra = [], set()
    da, db = ref / f"{route}.decisions.json", candidate / f"{route}.decisions.json"
    if da.exists() or db.exists():
        if not (da.exists() and db.exists()):
            problems.append({"where": "decisions.json", "reference": da.exists(), "candidate": db.exists()})
        else:
            a, b = json.loads(da.read_text()), json.loads(db.read_text())
            ca, cb = a.pop("chunks", []), b.pop("chunks", [])
            compare_records(a, b, "run", problems, extra)
            if len(ca) != len(cb):
                problems.append({"where": "chunks", "reference": len(ca), "candidate": len(cb)})
            for x, y in zip(ca, cb):
                compare_records(x, y, f"chunk{x.get('chunk')}", problems, extra)
            result["choices"] = [[c.get("chunk"), c.get("choice")] for c in cb if c.get("decided", "choice" in c)]
    ma, mb = ref / f"{route}.meta.json", candidate / f"{route}.meta.json"
    if ma.exists() and mb.exists():
        a, b = json.loads(ma.read_text()), json.loads(mb.read_text())
        for key in ("seed", "frames", "fps", "size", "crop", "prompt_sha256", "control_sha256", "history_chunks"):
            if key in a and key in b and a[key] != b[key]:
                problems.append({"where": f"meta.{key}", "reference": a[key], "candidate": b[key]})
    result["problems"] = problems[:40]
    result["problem_count"] = len(problems)
    result["keys_on_one_side"] = sorted(extra)
    result["identical"] = result["frames_equal"] and not problems
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--routes", nargs="+", required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    results = [compare_route(a.reference, a.candidate, r) for r in a.routes]
    ok = all(r["identical"] for r in results)
    text = json.dumps({"identical": ok, "routes": results}, indent=1, default=str)
    if a.out:
        a.out.write_text(text)
    print(text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
