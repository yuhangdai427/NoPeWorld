"""CPU tests of the head, its candidates, labels and losses, the operating-point rule, the released heads and the configs."""
from __future__ import annotations

import math
import os
from pathlib import Path
import unittest

import numpy as np
import torch

from nopeworld.config import METHODS, generator_config, generator_ids, method_available, method_config, modes_of
from nopeworld.head.candidates import DELTA, NAMES, RHO, STOP, KEEP, W_LATERAL, candidate_world_poses, intent_fit, path_length
from nopeworld.head.choose import choose
from nopeworld.head.labels import row_batch, sampling_weights
from nopeworld.head.losses import safety_losses
from nopeworld.head.model import FieldConfig, SafetyPolicy, load_head
from nopeworld.training.selection import working_point
from nopeworld.training.trainer import HeadSpec, init_seed, parse_head, spec_of

GENERATORS = ("lingbot_world_2", "hy_worldplay", "echo_wm", "evoke_turbo", "worldplay2", "hy_nomas")

# Table S3 of the paper: (T, tau, delta) per generator and mode, and the head's blocks.
OPERATING_POINTS = {
    ("lingbot_world_2", "collision"): ((30, 38, 39), 0.82, 0.85, 0.03), ("lingbot_world_2", "prediction"): ((30,), 0.906, 0.85, 0.01),
    ("hy_worldplay", "collision"): ((27,), 0.743, 0.80, 0.15), ("hy_worldplay", "prediction"): ((33,), 0.82, 0.80, 0.0),
    ("echo_wm", "collision"): ((42,), 0.906, 0.85, 0.0), ("echo_wm", "prediction"): ((30,), 1.0, 0.90, 0.02),
    ("evoke_turbo", "collision"): ((20,), 0.82, 0.80, 0.10),
    ("worldplay2", "collision"): ((30,), 0.82, 0.95, 0.03), ("worldplay2", "prediction"): ((20,), 0.82, 0.95, 0.075),
    ("hy_nomas", "collision"): ((33,), 0.906, 0.85, 0.0), ("hy_nomas", "prediction"): ((33,), 0.906, 0.85, 0.0),
}
# Table S1: parameters in millions (collision / prediction).
PARAMETERS = {"lingbot_world_2": (1.72, 1.74), "hy_worldplay": (0.95, 0.97), "echo_wm": (0.95, 0.97), "evoke_turbo": (1.07, None),
              "worldplay2": (1.72, 1.74), "hy_nomas": (0.70, 0.71)}


def weights_root():
    value = os.environ.get("NOPE_WEIGHTS")
    return Path(value) if value and (Path(value) / "heads").is_dir() else None


class Candidates(unittest.TestCase):
    def test_set(self):
        self.assertEqual(len(NAMES), 34)
        self.assertEqual((len(DELTA), len(RHO)), (34, 34))
        self.assertEqual((NAMES[KEEP], NAMES[STOP]), ("KEEP", "STOP"))
        self.assertEqual(RHO[STOP], 0.0)
        self.assertEqual(NAMES[2:6], ["SLOW750", "SLOW500", "SLOW250", "SLOW125"])
        self.assertEqual(RHO[2:6], [0.75, 0.5, 0.25, 0.125])
        self.assertEqual(sorted(DELTA[6:8]), [30.0, 60.0])

    def test_ellipse_lengths(self):
        """rho(alpha) = min(1, cos a / (cos^2 a + w sin^2 a)) at w = 0.5 on the 12 ellipse points of each side."""
        for name, delta, rho in zip(NAMES, DELTA, RHO):
            if name.startswith("EL"):
                a = math.radians(abs(delta))
                expected = min(1.0, math.cos(a) / (math.cos(a) ** 2 + W_LATERAL * math.sin(a) ** 2))
                self.assertAlmostEqual(rho, expected, places=3, msg=name)

    def test_intent_fit(self):
        a = intent_fit(DELTA, RHO)
        self.assertAlmostEqual(float(a[KEEP]), 1.0, places=6)
        self.assertAlmostEqual(float(a[STOP]), math.exp(-2), places=6)
        self.assertTrue(bool((a <= 1.0).all()))

    def test_candidate_poses(self):
        poses = np.tile(np.eye(4), (4, 1, 1))
        poses[:, 0, 3] = [0.0, 0.1, 0.2, 0.3]
        plen = path_length(poses, [0, 1, 2, 3])
        out = candidate_world_poses(poses, plen, first=False)
        self.assertEqual(out.shape, (34, 4, 4, 4))
        np.testing.assert_array_equal(out[KEEP], poses)
        np.testing.assert_allclose(out[STOP][:, :2, 3], np.zeros((4, 2)) + poses[0, :2, 3])
        np.testing.assert_allclose(np.linalg.norm(out[2, -1, :2, 3] - out[2, 0, :2, 3]), 0.75 * 0.3)


def fake_row(n=34, seed=0):
    rng = np.random.default_rng(seed)
    collision = rng.random(n)
    return {"id": f"s{seed}", "chunk": 3, "keep_index": 0, "collision_target": collision.tolist(),
            "policy_target": ((1 - collision) * intent_fit(DELTA, RHO).numpy()).tolist(),
            "candidate_rim_clearance_m": rng.uniform(-0.3, 1.2, n).tolist(), "candidate_observed": (rng.random(n) > 0.3).tolist(),
            "candidate_delta_deg": DELTA, "candidate_rho": RHO, "intent_length_m": 0.5, "candidate_A_slide": rng.random(n).tolist(),
            "stratum": "1-3s", "gt_status": "collide" if collision[0] > 0.5 else "clear"}


class LabelsAndLosses(unittest.TestCase):
    def test_row_batch(self):
        batch = row_batch(fake_row())
        for key in ("safe_target", "reward_target", "clearance_target", "observed", "intent_fit", "intent_target", "candidate_mask"):
            self.assertEqual(tuple(batch[key].shape), (1, 34), key)
        self.assertTrue(bool(batch["candidate_mask"].all()))

    def test_sampling_weights(self):
        rows = [fake_row(seed=i) | {"chunk": i % 5, "stratum": [">3s", "1-3s", "<1chunk", "contact"][i % 4]} for i in range(40)]
        w = sampling_weights(rows)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=6)
        self.assertTrue((w > 0).all())

    def test_losses(self):
        torch.manual_seed(0)
        batch = row_batch(fake_row())
        out = {"safe_logits": torch.randn(1, 34), "clearance": torch.randn(1, 34), "intent_logits": torch.randn(1, 34)}
        losses = safety_losses(out, batch)
        self.assertEqual(set(losses), {"total", "safe", "clearance", "listwise", "intent"})
        self.assertAlmostEqual(float(losses["total"]), float(losses["safe"] + 0.5 * losses["clearance"] + losses["listwise"] + losses["intent"]), places=5)
        for v in losses.values():
            self.assertTrue(math.isfinite(float(v)) and float(v) >= 0)


class Head(unittest.TestCase):
    def test_shapes_field(self):
        cfg = FieldConfig(layers=(1, 2), token_dim=32, camera_dim=6, frames=3, height=4, width=5, intent_head=True, camera_hidden=16)
        head = SafetyPolicy(cfg).eval()
        out = head(torch.randn(2, 2, 3, 4, 5, 32), torch.randn(2, 34, 3, 20, 6), torch.rand(2, 3).cumsum(-1))
        self.assertEqual({k: tuple(v.shape) for k, v in out.items()}, {"safe_logits": (2, 34), "clearance": (2, 34), "intent_logits": (2, 34)})

    def test_shapes_frame_with_command(self):
        cfg = FieldConfig(layers=(7,), token_dim=32, camera_dim=32, frames=1, height=4, width=5, intent_head=True, camera_tokens="frame",
                          camera_frames=4, command_input=True)
        head = SafetyPolicy(cfg).eval()
        mask = torch.ones(1, 34, dtype=torch.bool)
        mask[0, 5] = False
        out = head(torch.randn(1, 1, 1, 4, 5, 32), torch.randn(1, 34, 4, 32), torch.rand(1, 1), mask)
        self.assertEqual(tuple(out["safe_logits"].shape), (1, 34))
        self.assertEqual(float(out["safe_logits"][0, 5]), -math.inf)

    def test_head_names_and_seeds(self):
        self.assertEqual(parse_head("30+38+39:acc32,lr2").name, "L30+38+39_acc32_lr2")
        self.assertEqual(parse_head("20:commit").name, "L20_commit")
        self.assertEqual(parse_head("30:frozen").name, "L30_frozen")
        self.assertEqual(init_seed(parse_head("30:frozen"), 202), init_seed(HeadSpec(layers=(30,)), 202))
        for generator in GENERATORS:
            config = generator_config(generator)
            for mode, entry in config["heads"].items():
                self.assertEqual(spec_of(entry).name, entry["name"], (generator, mode))

    def test_choose_safe_first(self):
        logits = torch.full((1, 34), -5.0)
        a = intent_fit(DELTA, RHO)[None]
        mask = torch.ones(1, 34, dtype=torch.bool)
        self.assertEqual(int(choose(logits, a, mask, rule="safe_first", tau=0.8)["choice"][0]), STOP)
        logits[0, 0] = 5.0
        self.assertEqual(int(choose(logits, a, mask, rule="safe_first", tau=0.8)["choice"][0]), KEEP)
        logits[0, 0], logits[0, 8] = -5.0, 5.0
        self.assertEqual(int(choose(logits, a, mask, rule="safe_first", tau=0.8)["choice"][0]), 8)


@unittest.skipUnless(weights_root(), "needs NOPE_WEIGHTS with heads/")
class ReleasedHeads(unittest.TestCase):
    def test_operating_points_and_blocks(self):
        for (generator, mode), (blocks, t, tau, delta) in OPERATING_POINTS.items():
            head, payload = load_head(weights_root() / "heads" / generator / mode / "head.pt", "cpu")
            self.assertEqual(tuple(head.config.layers), blocks, (generator, mode))
            s = payload["selection"]
            self.assertEqual((s["rule"], round(s["temperature"], 3), s["tau"], s["keep_margin"]), ("safe_first", t, tau, delta), (generator, mode))
            self.assertEqual(head.config.command_input, mode == "prediction")
            self.assertEqual(payload["generator"], generator)

    def test_parameters(self):
        for generator, (collision, prediction) in PARAMETERS.items():
            for mode, millions in (("collision", collision), ("prediction", prediction)):
                path = weights_root() / "heads" / generator / mode / "head.pt"
                if millions is None:
                    self.assertFalse(path.exists())
                    continue
                head, _ = load_head(path, "cpu")
                self.assertEqual(round(sum(p.numel() for p in head.parameters()) / 1e6, 2), millions, (generator, mode))


@unittest.skipUnless(os.environ.get("NOPE_VAL_PREDS"), "needs NOPE_VAL_PREDS: a folder with <generator>.pt, the validation predictions of each released run")
class OperatingPointRule(unittest.TestCase):
    """The rule re-derives the released operating points from the validation predictions of the runs that trained the heads."""

    def test_reproduces_table(self):
        from nopeworld.training.selection import states_with_a_read
        root = Path(os.environ["NOPE_VAL_PREDS"])
        checked = 0
        for generator in GENERATORS:
            path = root / f"{generator}.pt"
            if not path.exists():
                continue
            preds = torch.load(path, map_location="cpu", weights_only=False)
            config = generator_config(generator)
            for mode, entry in config["heads"].items():
                _, t, tau, delta = OPERATING_POINTS[(generator, mode)]
                pred = states_with_a_read(preds[entry["name"]], entry.get("state", "step0"))
                point, within = working_point(pred, entry["missed_limit"])
                self.assertTrue(within, (generator, mode))
                self.assertEqual((round(point["temperature"], 3), point["tau"], point["keep_margin"]), (t, tau, delta), (generator, mode))
                checked += 1
        self.assertGreater(checked, 0)


class Configs(unittest.TestCase):
    def test_generators(self):
        self.assertEqual(sorted(generator_ids()), sorted(GENERATORS))
        for generator in GENERATORS:
            config = generator_config(generator)
            self.assertIn(config["fps"], (16, 24))
            self.assertIn(config["packages"], ("fps16", "fps24"))
            for value in config.get("paths", {}).values():
                self.assertFalse(str(value).startswith("/"), value)
            for mode, entry in config["heads"].items():
                self.assertIn(entry["missed_limit"], (0.05, 0.10))

    def test_methods(self):
        for method in METHODS:
            self.assertEqual(method_config(method)["id"], method)
        self.assertFalse(method_available(generator_config("evoke_turbo"), "nopeworld_prediction"))
        for generator in GENERATORS:
            for method in METHODS:
                expected = not (generator == "evoke_turbo" and method == "nopeworld_prediction")
                self.assertEqual(method_available(generator_config(generator), method), expected, (generator, method))
        self.assertEqual(modes_of(generator_config("evoke_turbo")), ["collision"])


if __name__ == "__main__":
    unittest.main()
