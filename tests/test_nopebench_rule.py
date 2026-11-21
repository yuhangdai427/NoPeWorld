"""Task 2: the NoPeBench rule (and its open-route form): stored verdicts re-judged from their signals, the noise floor and the map's vanish check
on made-up walks."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from nopebench.task2 import rule as R

FIXTURE = Path(__file__).parent / "fixtures" / "task2_rule_signals.json"
FLOAT_KEYS = ("progress", "sis_soft", "slide_m", "feasible_slide_m", "min_obstacle_distance_m", "noise_floor_m", "retreat_m", "past_m",
              "yaw_end_deg", "moved_last_m", "endpoint_progress", "max_step_m", "min_signed_distance_m")


def arrays(signals):
    out = {}
    for k, v in signals.items():
        if isinstance(v, list):
            a = np.array([np.nan if x is None else x for x in v] if v and not isinstance(v[0], list) else
                         [[np.nan if y is None else y for y in x] for x in v])
            out[k] = a.astype(bool) if a.dtype == object else a
        else:
            out[k] = v
    return out


def judge(item):
    sig, a = arrays(item["signals"]), item["args"]
    if a["control"]:
        return R.outcome_control(sig, np.asarray(a["u"]), a["speed"], noise_m=a["noise_m"])
    return R.outcome(sig, np.asarray(a["wall_normal"]), np.asarray(a["u"]), a["beta"], a["speed"], a["feasible"], noise_m=a["noise_m"])


ITEMS = json.loads(FIXTURE.read_text())["items"]


@pytest.mark.parametrize("item", ITEMS, ids=[i["kind"] for i in ITEMS])
def test_stored_verdicts(item):
    got = judge(item)
    for k, want in item["stored"].items():
        if k in FLOAT_KEYS and want is not None:
            assert got[k] == pytest.approx(want, abs=1.5e-3), k
        elif k in FLOAT_KEYS and want is None:
            assert got[k] is None or np.isnan(got[k]), k
        else:
            assert got[k] == want, k


def test_every_kind_is_covered():
    kinds = {i["kind"] for i in ITEMS}
    assert {"violation_through", "violation_vanished", "violation_past", "stopped", "slid", "early_stop", "retreated", "reversed",
            "uncertain_contact", "open_followed"} <= kinds


N = 40
U = np.array([0.0, 0.0, 1.0])
WALL = np.array([0.0, 0.0, -1.0])


def walk(stop_at=2.4):
    """Signals of a walk at 1.2 m/s straight at a wall 3 m ahead, standing from stop_at on (8 fps samples)."""
    t = np.arange(1, N + 1) / 8.0
    z = np.minimum(1.2 * t, stop_at)
    p = np.stack([np.zeros(N), np.zeros(N), z], 1)
    s = 3.0 - z
    ref = np.stack([np.zeros(81), np.zeros(81), np.minimum(np.arange(81) * 1.2 / 16, 2.75)], 1)
    return dict(t=t, p=p, s=s, s0=3.0, a=z.copy(), present=np.ones(N, bool), area=np.full(N, 0.1), border=np.zeros(N, bool), D=s.copy(),
                blocked=np.zeros(N, bool), near_frac=np.zeros(N), reference=ref, cmd_p=np.stack([np.zeros(N), np.zeros(N), 1.2 * t], 1),
                yaw_err=np.zeros(N), window_dir_prev=np.tile(U, (N, 1)), ahead_pred_all=np.zeros(N, bool), sd=s.copy(), sd_map=s.copy(),
                free_path=z.copy(), travelled=z.copy(), detour=np.full(N, -1.0), sd_pred=s.copy(), behind_pred=np.zeros(N, bool),
                ahead_map=np.zeros(N, bool))


def shallow(depth):
    sig = walk(stop_at=2.75)
    sig["sd"][-6:-4] = -depth                       # two samples in the geometry, at the end of the walk
    return sig


def test_within_the_floor_is_uncertain_contact():
    r = R.outcome(shallow(0.12), WALL, U, 0.0, 1.2, 0.0, noise_m=0.135)
    assert not r["violation"] and r["uncertain_contact"]
    assert (r["noise_floor_m"], r["inside_frames"]) == (0.135, 0)


def test_deeper_than_the_floor_is_a_violation():
    r = R.outcome(shallow(0.30), WALL, U, 0.0, 1.2, 0.0, noise_m=0.135)
    assert r["category"] == "violation_through" and not r["uncertain_contact"]


def test_floor_is_capped():
    assert R.noise_floor(2.0) == R.NOISE_CAP
    assert R.noise_floor(None) == R.TH2["inside_depth"]
    assert R.noise_floor(0.05) == R.TH2["inside_depth"]


def test_stop_at_the_wall():
    r = R.outcome(walk(stop_at=2.5), WALL, U, 0.0, 1.2, 0.0, noise_m=0.05)
    assert r["category"] == "stopped" and not r["violation"] and r["sis"]


def test_vanish_needs_the_map():
    sig = walk(stop_at=2.4)
    sig["D"] = np.full(N, 1.5)
    sig["D"][20:] = 6.0                             # a far depth jump
    sig["ahead_pred_all"][:] = True
    assert R.outcome(sig, WALL, U, 0.0, 1.2, 0.0)["category"] != "violation_vanished"
    sig["ahead_map"][:] = True
    assert R.outcome(sig, WALL, U, 0.0, 1.2, 0.0)["category"] == "violation_vanished"


def open_walk():
    """A walk along an open path, the reference being the command walked for the whole clip."""
    sig = walk(stop_at=10.0)
    sig.update(reference=np.stack([np.zeros(81), np.zeros(81), np.arange(81) * 1.2 / 16], 1), D=np.full(N, 8.0))
    for k in ("sd", "sd_pred", "sd_map"):
        sig[k] = np.full(N, 2.0)
    return sig


def test_open_route_follow():
    sig = open_walk()
    r = R.outcome_control(sig, U, 1.2, noise_m=0.05)
    assert r["category"] == "followed" and r["endpoint_progress"] == 1.0 and r["control_rule"] == "nopebench_open_route"
