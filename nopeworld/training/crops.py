"""Re-cuts the training crops (later starts of training episodes) that a generator's training rows name.

A crop `<episode_id>__cNNN` is the episode's decoded frames from `start16` (16 fps, `render/`) or `start24` (24 fps, `render24/`) on,
encoded once more with x264 at crf 10; `first_frame.png` is the decoded start frame, and the per-frame camera files are sliced the same
way. The parameters are in `manifest/train_crops.json`. With imageio-ffmpeg 0.6.0 (its Linux x86-64 ffmpeg 7.0.2, x264 core 164) the files
are byte for byte those the heads were trained on; another ffmpeg build can decode and encode them slightly differently.

    python -m nopeworld.training.crops --generator echo_wm --workers 2
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import time
import traceback

import numpy as np

from ..config import generator_ids
from ..paths import data_path, set_roots

CRF = 10
X264_THREADS = 3                     # part of the x264 bitstream: another thread count gives other bytes
FOLDERS = {"render": ("start16", 16), "render24": ("start24", 24)}
SLICED = ("intrinsics.npy", "intrinsics_native.npy")
CROP_VIDEO = re.compile(r'"source_video": "train/crops/([^/"]+)/([^/"]+)/clean\.mp4"')


def ffmpeg():
    """The ffmpeg binary of imageio-ffmpeg (IMAGEIO_FFMPEG_EXE when set, else the one it ships)."""
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def encode_command(width, height, fps, path):
    """ffmpeg arguments that encode raw RGB frames from stdin to H.264 at crf CRF."""
    return [ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(CRF), "-pix_fmt", "yuv420p", "-threads", str(X264_THREADS), "-f", "mp4", str(path)]


def write_mp4(frames, path, fps):
    """RGB uint8 [T, H, W, 3] to an H.264 mp4."""
    h, w = frames.shape[1:3]
    process = subprocess.Popen(encode_command(w, h, fps, path), stdin=subprocess.PIPE)
    process.stdin.write(memoryview(np.ascontiguousarray(frames)).cast("B"))
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {path}")


def crop_folders(generator):
    """{crop id: sub-folder} of the crops the generator's training rows name, in first-seen order."""
    found = {}
    with gzip.open(data_path("train", "rows", generator, "rows.jsonl.gz"), "rt") as fh:
        for line in fh:
            match = CROP_VIDEO.search(line)
            if match:
                crop, folder = match.groups()
                if found.setdefault(crop, folder) != folder:
                    raise ValueError(f"{generator}: rows of crop {crop} read both {found[crop]}/ and {folder}/")
    return found


def manifest_crops():
    """{crop id: crop} of manifest/train_crops.json."""
    return {c["id"]: c for c in json.loads(data_path("manifest", "train_crops.json").read_text())["crops"]}


def cut(crop, folder):
    """Writes train/crops/<id>/<folder>/; returns False when it is already there."""
    import cv2
    from ..utils.runtime import read_video
    start_key, fps = FOLDERS[folder]
    out = data_path("train", "crops", crop["id"], folder)
    if (out / "clean.mp4").exists():
        return False
    src = data_path("train", "episodes", crop["source"], folder)
    s = crop[start_key]
    frames = read_video(src / "clean.mp4")
    poses = np.load(src / "poses.npy")
    if len(poses) != len(frames):
        raise ValueError(f"{crop['id']}: {folder}/poses.npy has {len(poses)} poses for {len(frames)} frames")
    tmp = out.with_name(f".{folder}.{os.getpid()}.partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    write_mp4(frames[s:], tmp / "clean.mp4", fps)
    cv2.imwrite(str(tmp / "first_frame.png"), np.ascontiguousarray(frames[s][:, :, ::-1]))
    np.save(tmp / "poses.npy", poses[s:])
    for name in SLICED:
        if (src / name).exists():
            np.save(tmp / name, np.load(src / name)[s:])
    if (src / "timestamps.npy").exists():
        t = np.load(src / "timestamps.npy")
        np.save(tmp / "timestamps.npy", t[s:] - t[s])
    import imageio_ffmpeg
    written = imageio_ffmpeg.count_frames_and_secs(str(tmp / "clean.mp4"))[0]
    if written != len(frames) - s:
        raise ValueError(f"{crop['id']}: {folder} cut holds {written} frames, {len(frames) - s} written")
    if (out / "clean.mp4").exists():
        shutil.rmtree(tmp)
        return False
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return True


def cut_job(job):
    """cut() in a worker: its result or its error, never an exception."""
    crop, folder = job
    began = time.perf_counter()
    try:
        made = cut(crop, folder)
    except Exception as error:
        return {"id": crop["id"], "error": f"{type(error).__name__}: {error}", "trace": traceback.format_exc()}
    return {"id": crop["id"], "cut": made, "seconds": round(time.perf_counter() - began, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Re-cut the training crops a generator's training rows name (train/crops/<crop_id>/).")
    ap.add_argument("--generator", required=True, choices=generator_ids())
    ap.add_argument("--crops", nargs="+", help="only these crop ids (default: every crop of the generator's rows)")
    ap.add_argument("--workers", type=int, default=2, help="crops cut in parallel (each runs one ffmpeg with 3 x264 threads)")
    ap.add_argument("--data-root", help="dataset root (default $NOPE_DATA or ./data)")
    a = ap.parse_args(argv)
    set_roots(data=a.data_root)
    folders = crop_folders(a.generator)
    if a.crops:
        unknown = [c for c in a.crops if c not in folders]
        if unknown:
            ap.error(f"not crops of {a.generator}'s rows: {unknown[:5]}")
        folders = {c: folders[c] for c in a.crops}
    crops = manifest_crops()
    missing = [c for c in folders if c not in crops]
    if missing:
        raise SystemExit(f"crops of the rows not in manifest/train_crops.json: {missing[:5]}")
    jobs = [(crops[c], folder) for c, folder in folders.items() if not data_path("train", "crops", c, folder, "clean.mp4").exists()]
    print(json.dumps({"generator": a.generator, "crops": len(folders), "to_cut": len(jobs)}), flush=True)
    failed = []
    if jobs:
        with mp.get_context("spawn").Pool(min(a.workers, len(jobs))) as pool:
            for i, result in enumerate(pool.imap_unordered(cut_job, jobs)):
                if "error" in result:
                    failed.append(result)
                    print(result["trace"], flush=True)
                print(json.dumps({"done": i + 1, "of": len(jobs), **{k: v for k, v in result.items() if k != "trace"}}), flush=True)
    print(json.dumps({"cut": len(jobs) - len(failed), "failed": len(failed)}), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
