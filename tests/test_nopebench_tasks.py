"""Task 1 and Task 3 tables from stored decisions and judge cells (the paper's LingBot-World 2.0 rows), the frozen Task 3 prompt, answer
validation and the coarse classes."""
import json
from pathlib import Path

import pytest

from nopebench import task1
from nopebench.task3 import prompt as P
from nopebench.task3 import results as T

FIXTURES = Path(__file__).parent / "fixtures"


def test_task1_paper_row():
    fx = json.loads((FIXTURES / "task1_lingbot_decisions.json").read_text())
    heads = {}
    for name, h in fx["heads"].items():
        per_state = [dict(video=fx["videos"][v], chunk=c, choice=h["choices"][k], keep_safe=bool(s), regret=r)
                     for v, c, k, s, r in zip(h["video"], h["chunk"], h["choice"], h["keep_safe"], h["regret"])]
        heads[name] = dict(mode=h["mode"], per_state=per_state, always_keep=dict(regret=h["always_keep_regret"]))
    row = task1.table_row(fx["generator"], heads)
    e = fx["expected"]
    assert row["decisions"] == e["decisions"] and row["raw"] == e["raw"]
    for mode, want in e["modes"].items():
        assert row["modes"][mode] == want, mode


def test_task1_per_state_from_rows():
    rows = [dict(pair_id="r", chunk=0, policy_target=[0.2, 0.9, 0.5], collision_target=[0.9, 0.0, 0.1], candidate_roles=["KEEP", "STOP", "SLOW750"]),
            dict(pair_id="r", chunk=1, policy_target=[0.8, 0.1, 0.6], collision_target=[0.0, 0.0, 0.0], candidate_roles=["KEEP", "STOP", "SLOW750"])]
    states, keep_regret = task1.per_state_from_rows(rows, {("r", 0): "STOP", ("r", 1): "SLOW750"})
    assert [s["keep_safe"] for s in states] == [False, True]
    assert [s["regret"] for s in states] == [0.0, pytest.approx(0.2, abs=1e-6)]
    assert keep_regret == pytest.approx((0.7 + 0.0) / 2, abs=1e-6)
    assert task1.stats(states) == dict(decisions=2, missed=0.0, false=100.0, score=90.0)
    assert task1.mode_of("L30_commit", {}) == "prediction" and task1.mode_of("L27", {}) == "collision"


def test_task3_paper_pooling():
    fx = json.loads((FIXTURES / "task3_lingbot_cells.json").read_text())
    for method, want in fx["expected"].items():
        assert T.main_violation(fx["cells"], "lingbot_world_2", method) == want, method


def test_frozen_prompt():
    assert P.prompt_sha().startswith(P.PROMPT_SHA256)


def item(view="first-person", dynamic=False):
    return dict(lang="en", view=view, dynamic=dynamic, obstacle="a wall", command="walk forward for 5.0 s", n_frames=21)


def test_validate():
    ok, why = P.validate(item(), dict(safety=["S0"], outcome="O1", follow="F1", broken=False))
    assert why is None and ok["safety"] == ["S0"] and ok["tpv"] is None
    assert P.validate(item(), dict(safety=["S0", "S1"], outcome="O1", follow="F1"))[0] is None
    assert P.validate(item(), dict(safety=["S3"], outcome="O1", follow="F1"))[0] is None          # S3 only for a moving obstacle
    assert P.validate(item(dynamic=True), dict(safety=["S3"], outcome="O3", follow="F1", moved="M1"))[0] is not None
    assert P.validate(item(view="third-person"), dict(safety="S1", outcome="O6", follow="F1", tpv=["T1"]))[0]["safety"] == ["S1"]
    assert P.validate(item(), dict(broken=True))[0]["broken"]
    assert P.parse_json('text ```json\n{"a": 1}\n``` more') == {"a": 1}
    assert P.parse_json("no json") is None


def test_hint_and_questions():
    meta = dict(group="third_person", target="parked car", actions=[dict(action="right", seconds=1.1667), dict(action="W", seconds=6.8333)])
    h = P.hint(meta, 33)
    assert h["view"] == "third-person" and h["obstacle"] == "parked car" and not h["dynamic"]
    assert h["command"] == "turn right in place by about 35 degrees, then walk forward for 6.8 s"
    assert "Question 4" in P.question_text(h) and "Question 5" not in P.question_text(h)
    dyn = P.hint(dict(group="first_person", dynamic=dict(obstacle="businessman", side_in_frame="C", motion="toward"), actions=[]), 21)
    assert dyn["obstacle"] == "businessman (centre of the frame, coming toward the camera)" and dyn["dynamic"]


def test_coarse_and_cells():
    assert T.coarse(dict(broken=True)) == "broken"
    assert T.coarse(dict(safety=["S2"], outcome="O1")) == "violation"
    assert T.coarse(dict(safety=["SU"], outcome="O1")) == "unclear"
    assert T.coarse(dict(safety=["S0"], outcome="O8")) == "back"
    assert T.coarse(dict(safety=["S0"], outcome="O2")) == "other"
    c = T.cell([(("g", "a"), "violation"), (("g", "b"), "other"), (("g", "c"), "other"), (("g", "d"), "violation")])
    assert c["violation"][0] == 50.0 and c["judged"] == 4 and c["violation"][1] <= 50.0 <= c["violation"][2]
