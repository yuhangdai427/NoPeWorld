"""External depth baseline, the DA3 process's side: the labels on a map of depth points (walls of known distance), the server's answer with
a fake reconstruction, and its pipe protocol against the client."""
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nopeworld.head import occupancy as O
from nopeworld.inference import depth_server as S
from nopeworld.inference import depth_sweep as D
from nopeworld.inference.methods import external_depth as C

def wall_depth(distance, h=120, w=160, f=80.0):
    """A frontal wall `distance` m ahead of a camera at the origin (OpenCV), every pixel: depth = distance."""
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    return np.full((h, w), float(distance)), K


def test_map_distances():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-3, 3, (200, 2))
    m = D.depth_map(pts, np.zeros(2))
    q = rng.uniform(-2.5, 2.5, (300, 2))
    exact = np.sqrt(((q[:, None] - pts[None]) ** 2).sum(-1)).min(1)
    free = exact > 0.1
    np.testing.assert_allclose(m.sdf_at(q)[free], exact[free], atol=1.5 * D.VOXEL)
    assert float(D.depth_map(np.zeros((0, 2)), np.zeros(2)).sdf_at(np.zeros((1, 2)))[0]) > 5.0


def test_labels_at_a_wall():
    wall = np.stack([np.linspace(-4, 4, 1601), np.full(1601, 3.0)], 1)
    m = D.depth_map(wall, np.zeros(2))
    disps = np.array([[0.0, 2.9], [0.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
    h, P, A, slide = O.candidate_labels(m, np.zeros(2), disps, disps[0])
    np.testing.assert_allclose(h, [0.1 - 0.25, 3 - 0.25, 2 - 0.25, 3 - 0.25], atol=D.VOXEL)
    np.testing.assert_allclose(P, O.phi((h - O.PARAMS["m"]) / O.PARAMS["s"]), atol=1e-12)
    assert P[0] < 0.01 and P[2] > 0.99
    near = D.depth_map(np.stack([np.linspace(-4, 4, 1601), np.full(1601, 0.35)], 1), np.zeros(2))
    h, P, A, slide = O.candidate_labels(near, np.zeros(2), np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([0.0, 1.0]))
    assert np.linalg.norm(slide) < 0.15 and A[1] > 0.9 and A[0] < 0.3         # head-on: STOP fits the slide better than KEEP


def test_scores_from_depth():
    depth, K = wall_depth(3.0)
    keep, slow, side = [0.0, 0.0, 1.0], [0.0, 0.0, 0.5], [1.0, 0.0, 0.0]
    got, how = D.scores(depth[None], K[None], np.eye(4)[None], np.array([keep, slow, side]), tail=1, stride=1)
    np.testing.assert_allclose(got["h"], [2.0 - D.R_BODY, 2.5 - D.R_BODY, 3.0 - D.R_BODY], atol=D.VOXEL + 0.01)
    assert list(got["in_view"]) == [True, True, False] and how["band_points"] > 0
    with pytest.raises(ValueError):
        D.scores(depth[None], K[None], np.eye(4)[None], np.array([keep]), tail=2)


def test_band_and_level():
    pts = np.array([[0.0, 0.6, 1.0], [0.0, -0.3, 1.0], [0.0, 0.1, 2.0]])
    assert len(D.band(pts, 0.0)) == 1
    pitch = np.radians(-12.0)
    up = np.array([0.0, -np.cos(pitch), np.sin(pitch)])
    R = D.level_rotation(up)
    np.testing.assert_allclose(R @ up, [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(D.level_rotation([0, -1, 0]), np.eye(3), atol=1e-12)


def fake_reconstruct(depth, K, seen):
    def reconstruct(pictures, numbers, poses, kh):
        seen.update(pictures=len(pictures), numbers=list(numbers), kh=kh, poses=poses)
        n = len(numbers)
        return SimpleNamespace(scale=1.0, depth=np.stack([depth] * n), intrinsics=np.stack([K] * n), c2w=np.tile(np.eye(4), (n, 1, 1)),
                               history_fit={"position_error_max_m": 0.01}, frames=np.asarray(numbers), conf=None)
    return reconstruct


def test_answer_with_a_fake_reconstruction():
    depth, K = wall_depth(2.0)
    level = np.eye(4)
    level[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]                  # a level OpenCV camera in the poses' z-up world
    case = dict(kh=40, indices=np.array([32, 36, 40]), real=[np.zeros((4, 4, 3), np.uint8)] * 3, poses=np.tile(level, (41, 1, 1)))
    cases = SimpleNamespace(of=lambda request: case)
    seen = {}
    rec = fake_reconstruct(depth, K, seen)
    disps = [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    out = S.answer(dict(op="sweep", offsets=[2, 4, 6], disps_cam=disps, tail=2), np.zeros((3, 4, 4, 3), np.uint8), cases, rec)
    assert seen["numbers"] == [32, 36, 40, 42, 44, 46]
    np.testing.assert_allclose(out["h"], [1.0 - D.R_BODY, 2.0 - D.R_BODY], atol=D.VOXEL + 0.01)
    assert (len(out["P"]), len(out["A"]), out["in_view"]) == (2, 2, [True, False])
    out = S.answer(dict(op="sweep", offsets=[], disps_cam=disps, tail=2), None, cases, rec)      # the history alone
    assert (seen["numbers"], out["generated_frames"]) == ([32, 36, 40], 0)
    with pytest.raises(ValueError):
        S.answer(dict(op="sweep", offsets=[4, 2, 6], disps_cam=disps), np.zeros((3, 4, 4, 3), np.uint8), cases, rec)
    nan = lambda *a: SimpleNamespace(scale=float("nan"), history_fit={})
    out = S.answer(dict(op="sweep", offsets=[], disps_cam=disps), None, cases, nan)
    assert out["no_scale"] and out["P"] is None and out["h"] is None


def test_wbench_mode_takes_the_floor(tmp_path):
    cv2 = pytest.importorskip("cv2")
    cv2.imwrite(str(tmp_path / "handoff.png"), np.zeros((72, 128, 3), np.uint8))
    depth, K = wall_depth(2.0)
    seen = {}

    def floor(recon, origin, eye_m):
        seen.update(eye=eye_m, origin=origin)
        return 2.0, np.eye(3), dict(floor_frames=1)
    request = dict(op="sweep", mode="wbench", package=str(tmp_path), size=[64, 36], crop=None, offsets=[2, 4], disps_cam=[[0, 0, 1.0]], tail=1)
    out = S.answer(request, np.zeros((2, 36, 64, 3), np.uint8), None, fake_reconstruct(depth, K, seen), floor)
    assert (seen["pictures"], seen["poses"], seen["eye"], seen["origin"]) == (3, None, S.EYE_M, 0)
    assert out["h"][0] == pytest.approx(2 * 2.0 - 1.0 - D.R_BODY, abs=D.VOXEL + 0.01)          # the floor's scale 2: the wall at 4 m
    out = S.answer(request, np.zeros((2, 36, 64, 3), np.uint8), None, fake_reconstruct(depth, K, seen), lambda *a: (None, None, dict(floor_frames=0)))
    assert out["no_scale"]


def test_cases_are_keyed_by_package(tmp_path, monkeypatch):
    """Every route package of the dataset is named fps16: the history of one route must never serve another."""
    loaded = []
    cases = S.Cases()

    def fake_of(request):
        if cases.name == request["package"]:
            return cases.value
        loaded.append(request["package"])
        cases.name, cases.value = request["package"], dict(package=request["package"])
        return cases.value
    monkeypatch.setattr(cases, "of", fake_of)
    for pkg in ("a/fps16", "a/fps16", "b/fps16"):
        assert cases.of(dict(case="fps16", package=pkg))["package"] == pkg
    assert loaded == ["a/fps16", "b/fps16"]
    src = Path(S.__file__).read_text()
    assert 'self.name == request["package"]' in src and 'self.name = request["package"]' in src


def test_pipe_protocol_against_the_client():
    """The client's request bytes through serve(): one JSON line per answer, an error answered, end of input ends it."""
    frames = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
    requests = io.BytesIO()
    for op in ("sweep", "bad"):
        raw = frames.tobytes()
        requests.write((json.dumps(dict(op=op, offsets=[1, 2], bytes=len(raw), shape=list(frames.shape))) + "\n").encode() + raw)
    requests.seek(0)
    replies = io.BytesIO()

    def handle(request, got):
        if request["op"] != "sweep":
            raise ValueError("bad op")
        assert np.array_equal(got, frames)
        return dict(h=[1.0], P=[0.5], A=[1.0], in_view=[True])
    S.serve(requests, replies, handle)
    lines = replies.getvalue().decode().splitlines()
    assert json.loads(lines[0])["P"] == [0.5] and "error" in json.loads(lines[1]) and len(lines) == 2


def test_client_request_shape():
    class Fake:
        def __init__(self):
            self.asked = []

        def ask(self, request, frames):
            self.asked.append((request, frames.shape))
            n = len(request["disps_cam"])
            return dict(h=[0.5] * n, P=[0.9] * n, A=[0.8] * n, in_view=[True] * n, scale=1.0)
    fake = Fake()
    geo = C.GeoSafety(fake, Path("routes/r/fps16"), fps=16, mode="fpv")
    geo.size = [832, 464]
    geo.scores(1, 12, 12, np.zeros((34, 3)), lambda offs: np.zeros((len(offs), 4, 4, 3), np.uint8))
    request, shape = fake.asked[0]
    assert request["op"] == "sweep" and request["mode"] == "fpv" and request["tail"] == D.TAIL
    assert request["package"] == str(Path("routes/r/fps16")) and request["offsets"] == C.offsets_to_send(12, 12, 16)
