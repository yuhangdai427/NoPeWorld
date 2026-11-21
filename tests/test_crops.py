"""Training crops on the CPU: listing and cutting on a small synthetic episode."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True           # keep the source tree free of __pycache__

import contextlib
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from nopeworld.training import crops as NC

EPISODE = "Scene__fp001"
CROPS = [{"id": f"{EPISODE}__c004", "source": EPISODE, "scene": "Scene", "start16": 4, "start24": 6, "start_s": 0.25},
         {"id": f"{EPISODE}__c010", "source": EPISODE, "scene": "Scene", "start16": 10, "start24": 15, "start_s": 0.625}]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_frames(count, seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:48, :64]
    frames = [np.stack([(x * 4 + 3 * t) % 256, (y * 5 + 2 * t) % 256, (x + y + t) % 256], -1) for t in range(count)]
    noise = rng.integers(0, 24, (count, 48, 64, 3))
    return np.clip(np.stack(frames) + noise, 0, 255).astype(np.uint8)


def make_dataset(root):
    """A data root with one 16/24 fps episode, rows naming its two crops, and the crop manifest."""
    episode = root / "train" / "episodes" / EPISODE
    for folder, count in (("render", 30), ("render24", 45)):
        d = episode / folder
        d.mkdir(parents=True)
        NC.write_mp4(synthetic_frames(count, count), d / "clean.mp4", NC.FOLDERS[folder][1])
        poses = np.tile(np.eye(4), (count, 1, 1))
        poses[:, 0, 3] = np.arange(count) * 0.1
        np.save(d / "poses.npy", poses)
        if folder == "render":
            np.save(d / "intrinsics.npy", np.tile([416.0, 426.7, 416.0, 240.0], (count, 1)).astype(np.float32))
            np.save(d / "intrinsics_native.npy", np.tile([640.0, 640.0, 640.0, 360.0], (count, 1)))
            np.save(d / "timestamps.npy", np.arange(count) / 16.0 + 0.5)
    rows = root / "train" / "rows"
    for generator, folder in (("echo_wm", "render24"), ("worldplay2", "render")):
        (rows / generator).mkdir(parents=True)
        lines = [{"pair_id": EPISODE, "source_video": f"train/episodes/{EPISODE}/{folder}/clean.mp4", "chunk": 0}]
        lines += [{"pair_id": c["id"], "source_video": f"train/crops/{c['id']}/{folder}/clean.mp4", "chunk": k} for c in CROPS for k in range(2)]
        with gzip.open(rows / generator / "rows.jsonl.gz", "wt") as fh:
            fh.writelines(json.dumps(row) + "\n" for row in lines)
    (root / "manifest").mkdir()
    (root / "manifest" / "train_crops.json").write_text(json.dumps({"seed": 1, "passes": 1, "crops": CROPS}))


class Commands(unittest.TestCase):
    def test_encode_command(self):
        command = NC.encode_command(64, 48, 24, "out.mp4")
        joined = " ".join(command[1:])
        self.assertIn("-s 64x48 -r 24 -i -", joined)
        self.assertIn("-c:v libx264 -preset medium -crf 10 -pix_fmt yuv420p -threads 3 -f mp4 out.mp4", joined)

    def test_unknown_crop(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            root = Path(tmp)
            (root / "train" / "rows" / "echo_wm").mkdir(parents=True)
            with gzip.open(root / "train" / "rows" / "echo_wm" / "rows.jsonl.gz", "wt") as fh:
                fh.write(json.dumps({"source_video": "train/crops/a__c004/render24/clean.mp4"}) + "\n")
            with mock.patch.dict(os.environ, {"NOPE_DATA": tmp}), self.assertRaises(SystemExit) as stop:
                NC.main(["--generator", "echo_wm", "--crops", "b__c006"])
            self.assertEqual(stop.exception.code, 2)


@unittest.skipIf(cv2 is None, "needs opencv-python")
class Cut(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "data"
        make_dataset(self.root)
        self.env = mock.patch.dict(os.environ, {"NOPE_DATA": str(self.root)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_crop_folders(self):
        self.assertEqual(NC.crop_folders("echo_wm"), {c["id"]: "render24" for c in CROPS})
        self.assertEqual(NC.crop_folders("worldplay2"), {c["id"]: "render" for c in CROPS})
        with gzip.open(self.root / "train" / "rows" / "echo_wm" / "rows.jsonl.gz", "at") as fh:
            fh.write(json.dumps({"source_video": f"train/crops/{CROPS[0]['id']}/render/clean.mp4"}) + "\n")
        with self.assertRaises(ValueError):
            NC.crop_folders("echo_wm")

    def test_cut(self):
        from nopeworld.utils.runtime import read_video
        crop = CROPS[1]
        for folder in ("render", "render24"):
            s = crop[NC.FOLDERS[folder][0]]
            src = self.root / "train" / "episodes" / EPISODE / folder
            out = self.root / "train" / "crops" / crop["id"] / folder
            self.assertTrue(NC.cut(crop, folder))
            frames = read_video(src / "clean.mp4")
            cut = read_video(out / "clean.mp4")
            self.assertEqual(len(cut), len(frames) - s)
            np.testing.assert_array_equal(cv2.imread(str(out / "first_frame.png"))[:, :, ::-1], frames[s])
            np.testing.assert_array_equal(np.load(out / "poses.npy"), np.load(src / "poses.npy")[s:])
            self.assertFalse(NC.cut(crop, folder))
            self.assertFalse(any(p.name.startswith(".") for p in out.parent.iterdir()))
        render = self.root / "train" / "crops" / crop["id"] / "render"
        timestamps = np.load(render / "timestamps.npy")
        self.assertEqual(timestamps[0], 0.0)
        self.assertEqual(np.load(render / "intrinsics.npy").dtype, np.float32)
        self.assertEqual(sorted(p.name for p in (render.parent / "render24").iterdir()), ["clean.mp4", "first_frame.png", "poses.npy"])

    def test_main_skips_done(self):
        with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as stop:
            NC.main(["--generator", "worldplay2", "--crops", CROPS[0]["id"], "--workers", "1"])
        self.assertEqual(stop.exception.code, 0)
        self.assertTrue((self.root / "train" / "crops" / CROPS[0]["id"] / "render" / "clean.mp4").exists())
        with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
            NC.main(["--generator", "worldplay2", "--crops", CROPS[0]["id"]])
        self.assertEqual(json.loads(out.getvalue().splitlines()[0])["to_cut"], 0)


if __name__ == "__main__":
    unittest.main()
