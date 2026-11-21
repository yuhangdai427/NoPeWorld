"""Task 2 metrics: violation, back, SFP, slide, open-route SFP, success and the Safe-Intent Score (SIS), with scene-bootstrap intervals.

Everything is in the hand-off camera frame, horizontal plane (x, z), the hand-off position at the origin. One video of route r:
    p_end    per-axis median of the last three sampled camera positions after the hand-off (the evaluator samples 8 per second)
    r_end    the end of the route's Unreal Engine reference execution; d = |p_end - r_end|, D0 = |r_end|, eps = 0.3 m
    progress clip(1 - max(0, d - eps) / (D0 - eps), 0, 1); SFP is the progress, 0 with a violation
    slide    on the routes whose reference slides at least 0.5 m: the video's slide along the wall over the reference's, clipped to [0, 1],
             0 with a violation or going back
    success  the camera reaches the obstacle and stops or slides, or stops early within one chunk of travel of it
    SIS      gate * exp(-(u^2 + v^2) / (2 s^2)), s = 0.5: gate is 0 with a violation, a video against the input (back, wrong way, turned
             away, no response) or one that does not move; u the early-stop shortfall in chunks of travel at the generator's own speed
             (from its raw video), v = 1 - slide share on the slide routes
Rows are percent; intervals are percentiles 2.5 and 97.5 of 4000 draws of whole scenes (numpy default_rng(0)).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EPS = 0.30                  # m, the evaluator's position error
SLIDE_MIN = 0.5             # m: routes whose reference slides at least this far make the slide column
NEAR = 0.60                 # m, the evaluator's "at the obstacle"
DRAWS, SEED = 4000, 0
GEN_FRAMES = 80             # generated frames per route on the 16 fps clock (5 s)
BACK = ("reversed", "retreated")
AGAINST = ("reversed", "retreated", "wrong_direction", "turned_away", "no_response")
AT_OBSTACLE = ("slid", "stopped", "stopped_not_slid")
SIS_WIDTH = 0.5             # s of the SIS kernel, the width of the training labels' intent fit
FPS = 16.0
J_MAX = 24                  # frames: the window of the speed ratio is min(contact, 1.5 s)
CHUNK_S = {"lingbot_world_2": 0.75, "hy_worldplay": 2 / 3, "echo_wm": 1.0, "evoke_turbo": 1.5, "worldplay2": 1.0, "hy_nomas": 2 / 3}
S_UE = np.diag([1.0, -1.0, 1.0])


# ------------------------------------------------------------------------------------------------------------------- one video

def end_position(track_xz):
    """p_end: the per-axis median of the last three sampled positions after the hand-off ([T, 2], horizontal)."""
    p = np.asarray(track_xz, float).reshape(-1, 2)
    if not len(p):
        raise ValueError("no sampled position after the hand-off")
    return np.median(p[-3:], axis=0)


def progress(end_xz, ref_end_xz, eps=EPS):
    """(progress, d, D0) for a camera end position and the reference execution's end, both in the hand-off frame."""
    r = np.asarray(ref_end_xz, float)
    d0 = float(np.linalg.norm(r))
    if d0 <= eps:
        raise ValueError(f"degenerate route: the reference ends {d0:.3f} m from the hand-off, within eps = {eps} m")
    d = float(np.linalg.norm(np.asarray(end_xz, float) - r))
    return float(np.clip(1.0 - max(0.0, d - eps) / (d0 - eps), 0.0, 1.0)), d, d0


def slide_share(slide_m, reference_slide_m, violation, back):
    """Share of the reference's slide a video made; 0 with a violation or going back."""
    if violation or back:
        return 0.0
    return float(np.clip(slide_m or 0.0, 0.0, reference_slide_m) / reference_slide_m)


def score_video(route, record, track_xz, eps=EPS):
    """One video's row. record: the evaluator's verdict (None: no record); track_xz: the camera positions after the hand-off ([T, 2]).
    status: judged, unjudged (a record without a verdict) or missing (no record)."""
    out = dict(route_id=route["route_id"], scene=route["scene"], status="missing", category=None, violation=0.0, back=0.0, uncertain=0.0,
               d=None, D0=route["D0"], progress=0.0, sfp=0.0, slide=None, slide_m=None)
    slides = (not route["control"]) and route["ref_slide"] >= SLIDE_MIN
    if record is None:
        return out
    if not isinstance(record, dict) or "category" not in record:
        out.update(status="unjudged", slide=0.0 if slides else None)
        return out
    q, d, _ = progress(end_position(track_xz), route["ref_end"], eps)
    viol, back = bool(record["violation"]), record["category"] in BACK
    out.update(status="judged", category=record["category"], violation=float(viol), back=float(back),
               uncertain=float(bool(record.get("uncertain_contact"))), d=d, progress=q, sfp=0.0 if viol else q, slide_m=record.get("slide_m"),
               slide=slide_share(record.get("slide_m"), route["ref_slide"], viol, back) if slides else None)
    return out


def succeeds(record, generator, speed_mps):
    """Success: at the obstacle and stopped or slid, or an early stop within one chunk of travel of it."""
    if record["category"] in AT_OBSTACLE:
        return True
    return record["category"] == "early_stop" and (record.get("min_obstacle_distance_m") or 99.0) <= CHUNK_S[generator] * speed_mps


# ------------------------------------------------------------------------------------------------------------------- routes

def wall_in_camera(wall_xy, normal_xy, c2w):
    """The route's wall (point and normal, Unreal x-y) in the hand-off camera frame: horizontal normal towards the camera and its distance."""
    w = S_UE @ np.array([*wall_xy, 0.0])
    nrm = S_UE @ np.array([*normal_xy, 0.0])
    rot, c = c2w[:3, :3], c2w[:3, 3]
    p, n = rot.T @ (w - c), rot.T @ nrm
    n[1] = 0.0
    n /= np.linalg.norm(n)
    off = -n @ p
    return (-n, -off) if off < 0 else (n, off)


def slide_of(ref_rel, n_true, off_true, u, near=NEAR):
    """Slide (m) of the reference execution along the wall after it comes within `near` of it."""
    p = ref_rel[:, :3, 3]
    s = p @ n_true + off_true
    tangent = u - (u @ n_true) * n_true
    if np.linalg.norm(tangent) < 1e-6 or not (s <= near).any():
        return 0.0
    k = int(np.argmax(s <= near))
    return float((p[-1] - p[k]) @ (tangent / np.linalg.norm(tangent)))


def route_from_arrays(route_id, scene, reference_xz, *, control, contact=None, reference_slide_m=0.0, wall_normal=None, wall_offset=None,
                      version=None, speed_mps=None, handoff_frame=None):
    """A route record from its arrays: the reference execution in the hand-off frame ([T, 2]); for an obstacle route its contact frame,
    reference slide and wall plane (normal [3] towards the camera and offset, hand-off camera frame)."""
    ref = np.asarray(reference_xz, float).reshape(-1, 2)
    d0 = float(np.linalg.norm(ref[-1]))
    if d0 <= EPS:
        raise ValueError(f"{route_id}: degenerate route, the reference ends {d0:.3f} m from the hand-off")
    return dict(route_id=route_id, scene=scene, version=version, control=bool(control), kh=handoff_frame, reference=ref, ref_end=ref[-1], D0=d0,
                contact=None if control else int(contact), ref_slide=0.0 if control else float(reference_slide_m),
                wall_normal=None if wall_normal is None else np.asarray(wall_normal, float),
                wall_offset=None if wall_offset is None else float(wall_offset), speed_mps=speed_mps)


def load_route(package, route_id=None, scene=None, version=None):
    """What the metrics need of one 16 fps route package (meta.json, poses.npy, command.json, eval/meta.json, eval/reference_poses.npy)."""
    pkg = Path(package)
    meta = json.loads((pkg / "meta.json").read_text())
    ev = json.loads((pkg / "eval" / "meta.json").read_text())
    kh = int(meta["handoff_frame"])
    poses = np.load(pkg / "poses.npy").astype(float)
    inv = np.linalg.inv(poses[kh])
    ref = inv @ np.load(pkg / "eval" / "reference_poses.npy").astype(float)
    control = ev["kind"] == "control"
    extra = {}
    if not control:
        rel = inv @ poses[min(kh + 16, len(poses) - 1)]
        u = rel[:3, 3].copy()
        u[1] = 0.0
        u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([0.0, 0.0, 1.0])
        n_true, off_true = wall_in_camera(ev["wall_point_xy"], ev["wall_normal_xy"], poses[kh])
        extra = dict(contact=int(ev["event_frame_after_handoff"]), reference_slide_m=slide_of(ref, n_true, off_true, u),
                     wall_normal=n_true, wall_offset=off_true)
    command = pkg / "command.json"
    speed = float(json.loads(command.read_text())["speed_mps"]) if command.is_file() else ev.get("speed_mps")
    route_id = route_id or meta.get("case") or pkg.name
    return route_from_arrays(route_id, scene or meta.get("scene") or route_id.split("__")[0], ref[:, :3, 3][:, [0, 2]], control=control,
                             version=version, speed_mps=speed, handoff_frame=kh, **extra)


def wall_distance(route):
    """g(p): the wall-normal distance of a horizontal position (x, z) in the hand-off frame."""
    n, off = route["wall_normal"], route["wall_offset"]
    return lambda p: float(np.array([p[0], 0.0, p[1]]) @ n + off)


def perfect_stop(route, eps=EPS):
    """Progress of the perfect stop: the reference execution up to its contact frame, then standing (the reference itself on an open route)."""
    if route["control"]:
        return 1.0
    k = min(route["contact"], len(route["reference"]) - 1)
    return progress(route["reference"][k], route["ref_end"], eps)[0]


# ------------------------------------------------------------------------------------------------------------------- videos

def track_of(measurement, handoff_frame):
    """A video's camera track after the hand-off from its saved measurement: dict(frames after the hand-off, xz [T, 2])."""
    with np.load(measurement) as z:
        frames, c2w = np.asarray(z["frames"]), np.asarray(z["c2w"], float)
    gen = frames > handoff_frame
    return dict(frames=frames[gen] - handoff_frame, xz=c2w[gen][:, :3, 3][:, [0, 2]])


def read_videos(eval_dir, routes, recon="vggt_omega"):
    """{route_id: dict(record, frames, xz)} of one evaluated run (verdicts.json and measure/<route_id>.npz under eval_dir); routes with
    neither are left out. Every verdict must come from the `recon` backend."""
    eval_dir = Path(eval_dir)
    verdicts = json.loads((eval_dir / "verdicts.json").read_text()) if (eval_dir / "verdicts.json").is_file() else {}
    out = {}
    for r in routes:
        rec = verdicts.get(r["route_id"])
        npz = eval_dir / "measure" / f"{r['route_id']}.npz"
        if rec is None and not npz.is_file():
            continue
        if isinstance(rec, dict) and "category" in rec:
            if rec.get("recon") != recon:
                raise ValueError(f"{eval_dir}: {r['route_id']} was judged on {rec.get('recon')}, not {recon}")
            if not npz.is_file():
                raise FileNotFoundError(f"{npz}: a verdict without its measurement")
        video = dict(record=rec, frames=np.zeros(0, int), xz=np.zeros((0, 2)))
        if npz.is_file():
            video.update(track_of(npz, r["kh"]))
        out[r["route_id"]] = video
    return out


def judged(video):
    return video is not None and isinstance(video.get("record"), dict) and "category" in video["record"]


# ------------------------------------------------------------------------------------------------------------------- statistics

class Resampler:
    """Whole scenes drawn with replacement: `n` draws of as many scenes as there are; draw i picks rng.integers(0, k, k) from the sorted
    scene names. One object gives every column the same draws."""

    def __init__(self, scenes, n=DRAWS, seed=SEED):
        self.names = sorted(set(scenes))
        self.index = {s: i for i, s in enumerate(self.names)}
        k = len(self.names)
        picks = np.random.default_rng(seed).integers(0, k, size=(n, k))
        self.weight = np.zeros((n, k))
        np.add.at(self.weight, (np.repeat(np.arange(n), k), picks.ravel()), 1.0)

    def groups(self, scenes):
        return np.array([self.index[s] for s in scenes], int)

    def sums(self, values, scenes):
        """Per draw: the sum of `values` over the drawn scenes (a scene drawn twice counts twice)."""
        v = np.asarray(values, float)
        return self.weight @ np.bincount(self.groups(scenes), weights=v, minlength=len(self.names))

    def mean_draws(self, values, scenes):
        num, den = self.sums(values, scenes), self.sums(np.ones(len(values)), scenes)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    def ratio_draws(self, num, den, scenes):
        a, b = self.sums(num, scenes), self.sums(den, scenes)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def interval(point, draws, n, scale=100.0, digits=2):
    """A value with its 95 % interval (percentiles 2.5 and 97.5 of the draws), times `scale`."""
    if n == 0 or point is None:
        return dict(n=int(n), value=None, ci95=None)
    lo, hi = np.nanpercentile(np.asarray(draws, float), [2.5, 97.5])
    return dict(n=int(n), value=round(float(point) * scale, digits), ci95=[round(float(lo) * scale, digits), round(float(hi) * scale, digits)])


def mean_interval(rs, values, scenes):
    v = np.asarray(values, float)
    return interval(v.mean() if len(v) else None, rs.mean_draws(v, scenes) if len(v) else [], len(v))


def columns(rs, rows):
    """The columns of one method on one set of routes, from its per-route rows."""
    scenes = [r["scene"] for r in rows]
    out = dict(routes=len(rows), unjudged=sum(r["status"] == "unjudged" for r in rows), uncertain_contact=int(sum(r["uncertain"] for r in rows)))
    for key in ("violation", "back", "sfp"):
        out[key] = mean_interval(rs, [r[key] for r in rows], scenes)
    sl = [r for r in rows if r["slide"] is not None]
    out["slide"] = mean_interval(rs, [r["slide"] for r in sl], [r["scene"] for r in sl])
    return out


def paired(rs, rows_a, rows_b):
    """a - b on the same routes, the same draws."""
    assert [r["route_id"] for r in rows_a] == [r["route_id"] for r in rows_b]
    scenes = [r["scene"] for r in rows_a]
    out = {key: mean_interval(rs, [x[key] - y[key] for x, y in zip(rows_a, rows_b)], scenes) for key in ("violation", "back", "sfp")}
    sl = [(x, y) for x, y in zip(rows_a, rows_b) if x["slide"] is not None]
    out["slide"] = mean_interval(rs, [x["slide"] - y["slide"] for x, y in sl], [x["scene"] for x, _ in sl])
    return out


def scene_bootstrap(values, scenes, draws=DRAWS, seed=SEED):
    """The SIS interval: mean x 100 (one decimal) and percentiles 2.5 / 97.5 of `draws` draws of whole scenes, one rng.integers call each."""
    names = sorted(set(scenes))
    sc = np.array([names.index(s) for s in scenes])
    rng = np.random.default_rng(seed)
    picks = [rng.integers(0, len(names), len(names)) for _ in range(draws)]
    v = np.asarray(values, float)
    sums, cnt = np.bincount(sc, v, len(names)), np.bincount(sc, None, len(names))
    m = [sums[d].sum() / cnt[d].sum() for d in picks]
    return dict(value=round(100 * float(v.mean()), 1), ci95=[round(100 * float(np.percentile(m, 2.5)), 1), round(100 * float(np.percentile(m, 97.5)), 1)])


# ------------------------------------------------------------------------------------------------------------------- SIS

def sis_value(gate, u, v, s=SIS_WIDTH):
    """The score of one video from its gate, u and v."""
    return 0.0 if not gate else float(np.exp(-(u * u + v * v) / (2 * s * s)))


def closest_gap(xz, g):
    """The camera's wall-normal distance at its closest approach: the median of its three smallest values."""
    return float(np.median(np.sort([g(p) for p in xz])[:3]))


def at(ref, t):
    """Reference position at fractional frame t (linear interpolation, clamped to the clip)."""
    t = float(np.clip(t, 0.0, len(ref) - 1))
    i = int(np.floor(t))
    return ref[i] if i + 1 >= len(ref) else ref[i] * (1 - (t - i)) + ref[i + 1] * (t - i)


def speed_ratio(route, video):
    """rho: the raw video's displacement over the reference's at frame min(contact, 1.5 s) (at least 8), clipped to [0.3, 2]."""
    ref = route["reference"]
    j_w = max(8, min(route["contact"], J_MAX))
    fr, xz = np.asarray(video["frames"]), np.asarray(video["xz"], float)
    if not len(fr) or fr.max() < j_w:
        return None
    k = int(np.argmin(np.abs(fr - j_w)))
    den = float(np.linalg.norm(ref[int(fr[k])]))
    if den < 0.15:
        return None
    return float(np.clip(np.linalg.norm(xz[k]) / den, 0.3, 2.0))


def windows(routes, raw_videos, generator):
    """{route_id: (g_c, L, rho)}: the reference's wall distance at contact, how much it closes on the wall in one chunk before contact at
    the generator's own speed rho (from its raw videos; the median where a route's raw video gives none)."""
    rhos = {r["route_id"]: speed_ratio(r, raw_videos[r["route_id"]]) for r in routes
            if raw_videos.get(r["route_id"]) is not None and len(raw_videos[r["route_id"]]["frames"])}
    med = float(np.median([v for v in rhos.values() if v is not None]))
    T = CHUNK_S[generator] * FPS
    out = {}
    for r in routes:
        rho = rhos.get(r["route_id"]) or med
        ref, c = r["reference"], r["contact"]
        g = wall_distance(r)
        gc = g(ref[c])
        out[r["route_id"]] = (gc, max(g(at(ref, c - rho * T)) - gc, 0.05), rho)
    return out


def sis_rows(routes, videos, raw_videos, generator, metric_rows):
    """Per-route SIS of one method. metric_rows: its score_video rows (the slide share, as the per-route table stores it: 4 decimals)."""
    share = {m["route_id"]: (None if m["slide"] is None else round(m["slide"], 4)) for m in metric_rows}
    win = windows(routes, raw_videos, generator)
    out = []
    for r in routes:
        gc, L, rho = win[r["route_id"]]
        video = videos.get(r["route_id"])
        if not judged(video):
            out.append(dict(route_id=r["route_id"], scene=r["scene"], status="missing", gate=0, u=None, v=None, sis=0.0))
            continue
        xz, rec = np.asarray(video["xz"], float).reshape(-1, 2), video["record"]
        cat = rec["category"]
        moved = len(xz) > 0 and float(np.linalg.norm(xz, axis=1).max()) >= 0.1
        gate = int(not bool(rec["violation"]) and cat not in AGAINST and moved)
        gap = closest_gap(xz, wall_distance(r)) if len(xz) else float("nan")
        u = max(0.0, gap - gc - EPS) / L if len(xz) else float("nan")
        sh = share.get(r["route_id"])
        v = 0.0 if sh is None else 1.0 - sh
        val = sis_value(gate, u, v)
        out.append(dict(route_id=r["route_id"], scene=r["scene"], status="judged", category=cat, gate=gate, gap_m=round(gap, 3), g_contact_m=round(gc, 3),
                        chunk_close_m=round(L, 3), rho=round(rho, 3), u=round(u, 3), v=round(v, 3), sis=val))
    return out


# ------------------------------------------------------------------------------------------------------------------- one method

def method_rows(routes, videos):
    """score_video rows of one method on a set of routes."""
    rows = []
    for r in routes:
        video = videos.get(r["route_id"])
        if video is None or video.get("record") is None:
            rows.append(score_video(r, None, None))
        else:
            rows.append(score_video(r, video["record"], video["xz"] if judged(video) else None))
    return rows


def task2_table(routes, videos, *, generator, raw_videos=None, open_routes=None, open_videos=None, draws=DRAWS, seed=SEED):
    """Every Task 2 column of one generator x method from read_videos() dicts. raw_videos: the same generator's raw run, whose speed sets the
    SIS window. Returns (table, per-route rows); a method without a record on some route gets no numbers (incomplete)."""
    rows = method_rows(routes, videos)
    missing = [r["route_id"] for r in rows if r["status"] == "missing"]
    out = dict(generator=generator, routes=len(routes), incomplete=bool(missing), missing=missing)
    per_route = dict(routes=rows, sis=None, open=None)
    if missing:
        return out, per_route
    rs = Resampler([r["scene"] for r in routes], draws, seed)
    out.update(columns(rs, rows))
    out["success"] = mean_interval(rs, [float(judged(videos[r["route_id"]]) and succeeds(videos[r["route_id"]]["record"], generator, r["speed_mps"]))
                                        for r in routes], [r["scene"] for r in routes])
    out["perfect_stop_sfp"] = mean_interval(rs, [perfect_stop(r) for r in routes], [r["scene"] for r in routes])
    if raw_videos is not None:
        sis = sis_rows(routes, videos, raw_videos, generator, rows)
        out["sis"] = scene_bootstrap([s["sis"] for s in sis], [s["scene"] for s in sis], draws, seed)
        out["sis_gate"] = scene_bootstrap([s["gate"] for s in sis], [s["scene"] for s in sis], draws, seed)
        per_route["sis"] = sis
    if open_routes:
        k_rows = method_rows(open_routes, open_videos or {})
        k_missing = [r["route_id"] for r in k_rows if r["status"] == "missing"]
        out["open"] = dict(routes=len(open_routes), incomplete=bool(k_missing), missing=k_missing)
        if not k_missing:
            kc = columns(Resampler([r["scene"] for r in open_routes], draws, seed), k_rows)
            out["open"].update({k: kc[k] for k in ("unjudged", "uncertain_contact", "violation", "back", "sfp")})
        per_route["open"] = k_rows
    return out, per_route
