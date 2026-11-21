"""Task 2 metrics: the paper's LingBot-World 2.0 rows recomputed from its stored verdicts and camera tracks (tests/fixtures)."""
import json
from pathlib import Path

import numpy as np
import pytest

from nopebench.task2 import metrics as M

FIXTURE = Path(__file__).parent / "fixtures" / "task2_lingbot_metrics.json"
# the main table: SIS, Viol. (%), Slide (%)
PAPER = {"raw": (4.7, 89.0, 3.78), "nopeworld_collision": (50.0, 7.0, 18.86), "nopeworld_prediction": (55.6, 4.0, 29.8)}


@pytest.fixture(scope="module")
def data():
    fx = json.loads(FIXTURE.read_text())
    routes = [M.route_from_arrays(r["route_id"], r["scene"], r["reference_xz"], control=False, contact=r["contact"],
                                  reference_slide_m=r["reference_slide_m"], wall_normal=r["wall_normal"], wall_offset=r["wall_offset"],
                                  version=r["version"], speed_mps=r["speed_mps"]) for r in fx["routes"]]
    open_routes = [M.route_from_arrays(r["route_id"], r["scene"], r["reference_xz"], control=True, speed_mps=r["speed_mps"]) for r in fx["open_routes"]]
    frames = np.asarray(fx["frames_after_handoff"])

    def videos(part):
        return {m: {c: dict(record=v["record"], frames=frames[-len(v["xz"]):], xz=np.asarray(v["xz"])) for c, v in vs.items()} for m, vs in fx[part].items()}
    vids, open_vids = videos("videos"), videos("open_videos")
    tables = {m: M.task2_table(routes, vids[m], generator=fx["generator"], raw_videos=vids["raw"], open_routes=open_routes, open_videos=open_vids[m])
              for m in PAPER}
    return fx, routes, tables


@pytest.mark.parametrize("method", list(PAPER))
def test_paper_numbers(data, method):
    _, _, tables = data
    t, _ = tables[method]
    sis, viol, slide = PAPER[method]
    assert not t["incomplete"]
    assert t["sis"]["value"] == sis
    assert round(t["violation"]["value"]) == viol
    assert t["slide"]["value"] == slide
    assert t["slide"]["n"] == 65 and t["routes"] == 100


@pytest.mark.parametrize("method", list(PAPER))
def test_stored_intervals(data, method):
    fx, _, tables = data
    t, _ = tables[method]
    e = fx["expected"][method]
    assert t["sis"] == e["sis"]
    for key in ("violation", "slide", "back", "sfp"):
        assert {k: t[key][k] for k in ("value", "ci95")} == e[key], key
    assert {k: t["open"]["sfp"][k] for k in ("value", "ci95")} == e["open_sfp"]
    assert t["success"]["value"] == pytest.approx(100 * e["success"], abs=1e-9)


@pytest.mark.parametrize("method", list(PAPER))
def test_per_route(data, method):
    """Per route to 1e-3: the fixture's positions are rounded to 0.1 mm."""
    fx, _, tables = data
    t, per_route = tables[method]
    e = fx["expected"][method]
    for s in per_route["sis"]:
        assert s["sis"] == pytest.approx(e["per_route_sis"][s["route_id"]], abs=1e-3), s["route_id"]
    for r in per_route["routes"]:
        assert r["sfp"] == pytest.approx(e["per_route_sfp"][r["route_id"]], abs=1e-3), r["route_id"]
        if r["slide"] is not None:
            assert r["slide"] == pytest.approx(e["per_route_slide"][r["route_id"]], abs=1e-3), r["route_id"]


def test_progress_and_slide_share():
    assert M.progress([0.0, 2.0], [0.0, 2.0])[0] == 1.0
    assert M.progress([0.0, 0.0], [0.0, 2.0])[0] == 0.0
    assert M.progress([0.0, 1.0], [0.0, 2.0])[0] == pytest.approx(1 - 0.7 / 1.7)
    assert M.slide_share(0.5, 1.0, False, False) == 0.5
    assert M.slide_share(2.0, 1.0, False, False) == 1.0
    assert M.slide_share(0.5, 1.0, True, False) == 0.0
    assert M.slide_share(0.5, 1.0, False, True) == 0.0
    with pytest.raises(ValueError):
        M.progress([0, 0], [0.0, 0.2])


def test_sis_value():
    assert M.sis_value(0, 0.0, 0.0) == 0.0
    assert M.sis_value(1, 0.0, 0.0) == 1.0
    assert M.sis_value(1, 0.5, 0.0) == pytest.approx(np.exp(-0.5))


def test_incomplete_method_has_no_numbers(data):
    fx, routes, _ = data
    frames = np.asarray(fx["frames_after_handoff"])
    vids = {c: dict(record=v["record"], frames=frames, xz=np.asarray(v["xz"])) for c, v in fx["videos"]["raw"].items()}
    vids.pop(routes[0]["route_id"])
    t, _ = M.task2_table(routes, vids, generator=fx["generator"], raw_videos=vids)
    assert t["incomplete"] and t["missing"] == [routes[0]["route_id"]] and "sis" not in t


def test_resampler_draws_are_whole_scenes():
    rs = M.Resampler(["a", "a", "b"], n=50, seed=0)
    assert rs.weight.sum(1).tolist() == [2.0] * 50
    draws = rs.mean_draws([1.0, 1.0, 0.0], ["a", "a", "b"])
    assert set(np.round(draws, 6)) <= {0.0, round(2 / 3, 6), 1.0}
