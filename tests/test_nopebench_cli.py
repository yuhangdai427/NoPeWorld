"""nopebench's console scripts and dataset readers on a tiny dataset of the published layout."""
import json

import numpy as np
import pytest

from nopebench import cli, data


def make_route(root, route_id, scene="Scene", kind="critical"):
    """A route folder of the dataset layout with a straight walk 3 m at a wall."""
    d = root / data.TASK2_ROUTES / route_id
    pkg = d / "fps16"
    (pkg / "eval").mkdir(parents=True)
    kh = 10
    poses = np.tile(np.eye(4), (kh + 81, 1, 1))
    poses[:, :3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]           # a level camera looking along the world's y (z up)
    poses[:, 1, 3] = np.arange(kh + 81) * 0.075
    ref = poses[kh:].copy()
    ref[:, 1, 3] = np.minimum(ref[:, 1, 3], poses[kh, 1, 3] + 2.0)
    np.save(pkg / "poses.npy", poses)
    np.save(pkg / "eval" / "reference_poses.npy", ref)
    (pkg / "meta.json").write_text(json.dumps(dict(handoff_frame=kh, valid=True)))
    (pkg / "command.json").write_text(json.dumps(dict(speed_mps=1.2)))
    (pkg / "eval" / "meta.json").write_text(json.dumps(dict(kind=kind, event_frame_after_handoff=30, wall_point_xy=[0.0, -(kh * 0.075 + 3.0)], wall_normal_xy=[0.0, 1.0])))
    (d / data.ROUTE_JSON).write_text(json.dumps(dict(route_id=route_id, scene=scene, version="original")))
    return d


def test_readers(tmp_path, monkeypatch):
    monkeypatch.setenv("NOPE_DATA", str(tmp_path))
    (tmp_path / data.MANIFEST).mkdir()
    (tmp_path / data.MANIFEST / data.ROUTE_LISTS["task2"]).write_text("Scene__n001\nScene__n002\n")
    for r in ("Scene__n001", "Scene__n002"):
        make_route(tmp_path, r)
    assert data.route_ids("task2") == ["Scene__n001", "Scene__n002"]
    assert data.route_package("Scene__n001", 24) == tmp_path / "task2/routes/Scene__n001/fps24"
    assert data.route_dir("X", "task2_open") == tmp_path / "task2/open_routes/X"
    routes = data.task2_routes("task2")
    assert [r["route_id"] for r in routes] == ["Scene__n001", "Scene__n002"]
    assert routes[0]["scene"] == "Scene" and routes[0]["contact"] == 30 and routes[0]["speed_mps"] == 1.2
    assert routes[0]["D0"] == pytest.approx(2.0)
    assert routes[0]["wall_offset"] == pytest.approx(3.0) and abs(routes[0]["wall_normal"] @ [0, 0, 1.0]) == pytest.approx(1.0)


def test_jsonl_gz(tmp_path):
    import gzip
    with gzip.open(tmp_path / "rows.jsonl.gz", "wt") as fh:
        fh.write('{"a": 1}\n{"a": 2}\n')
    assert [r["a"] for r in data.read_jsonl(tmp_path / "rows.jsonl.gz")] == [1, 2]


def test_eval_arguments(tmp_path):
    with pytest.raises(SystemExit):
        cli.eval_main(["--task", "task2", "--run", str(tmp_path)])          # no generator, method or run.json
    (tmp_path / "task1_report.json").write_text(json.dumps(dict(heads={"L27": dict(per_state=[
        dict(video="r", chunk=0, choice="KEEP", keep_safe=True, regret=0.0), dict(video="r", chunk=1, choice="STOP", keep_safe=False, regret=0.1)],
        always_keep=dict(regret=0.3))})))
    cli.eval_main(["--task", "task1", "--run", str(tmp_path), "--generator", "hy_worldplay"])
    row = json.loads((tmp_path / "task1_table.json").read_text())
    assert row["modes"]["collision"]["own"] == dict(decisions=2, missed=0.0, false=0.0, score=95.0) and row["raw"]["score"] == 70.0


def test_task1_heads_from_two_runs(tmp_path):
    state = lambda chunk, choice, safe, regret: dict(video="r", chunk=chunk, choice=choice, keep_safe=safe, regret=regret)
    runs = {"collision": ("L27", [state(0, "KEEP", True, 0.0), state(1, "STOP", False, 0.1)]),
            "prediction": ("L33_commit", [state(1, "KEEP", False, 0.4)])}
    for mode, (name, states) in runs.items():
        (tmp_path / mode).mkdir()
        (tmp_path / mode / "task1_report.json").write_text(json.dumps(dict(heads={name: dict(mode=mode, per_state=states,
                                                                                               always_keep=dict(regret=0.4))})))
    cli.eval_main(["--task", "task1", "--runs", str(tmp_path / "collision"), str(tmp_path / "prediction"), "--generator", "hy_worldplay",
                   "--out", str(tmp_path / "table.json")])
    row = json.loads((tmp_path / "table.json").read_text())
    assert row["decisions"] == 1 and row["raw"]["score"] == 60.0
    assert row["modes"]["collision"]["common"] == dict(decisions=1, missed=0.0, false=None, score=90.0)
    assert row["modes"]["prediction"]["common"] == dict(decisions=1, missed=100.0, false=None, score=60.0)
    with pytest.raises(SystemExit):
        cli.eval_main(["--task", "task2", "--runs", str(tmp_path / "collision"), "--generator", "hy_worldplay", "--method", "raw"])


def test_both_heads_is_task1_only(tmp_path):
    from nopeworld import cli as nw
    with pytest.raises(SystemExit):
        nw.infer_main(["--generator", "hy_worldplay", "--method", "nopeworld", "--task", "task2", "--out", str(tmp_path)])
