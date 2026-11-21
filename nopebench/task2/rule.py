"""The NoPeBench rule of Task 2 (and its open-route form): per-frame signals in the hand-off camera frame and the verdict of one video.

World frame = the hand-off camera (OpenCV: x right, y down, z forward; metres). Distances are horizontal (x, z). The collision volume is
the first-person camera's head and shoulders: a vertical cylinder of r = 0.25 m from 0.5 m below the eye to 0.2 m above it.

A violation is the camera visibly going into or through geometry, or the obstacle vanishing:
  inside    the camera centre deeper in the true scene geometry (task2.scene.SceneField) than the video's own history fit error (the
            reconstruction's position error on the real history frames), clipped to [0.10 m, 0.30 m], for at least two consecutive
            sampled frames (8 fps). A run deeper than 0.10 m but within that floor is flagged uncertain_contact instead.
  behind    the centre in free space at a place whose free-space path from the hand-off camera is longer than 1.1 x the camera's own
            path + 0.3 m: it went through a thin wall.
  jump      the depth ahead (20th percentile of the window around the commanded direction) jumps within one step to >= 3x and >= 2 m:
            through the obstacle when it was at most 0.8 m before and geometry is ahead (the hand-off depth's points, the map's rays, or
            inside / behind at the extrapolated pose); the obstacle vanishing when it was further and the map agrees it should be seen.
  tracker   the obstacle lost by the tracker while it is ahead, in view and not at the border, where the map agrees it should be seen.
  past      the centre came to the true geometry and then got 1.5 m (or 0.375 s of travel) further along the wall's inward normal than
            the reference execution ever gets.
Categories after the violations: no_response, reversed, retreated, wrong_direction, early_stop / still_moving (never at the obstacle),
turned_away, slid, stopped, stopped_not_slid. On an open route (no obstacle): followed, early_stop and the same departures.
"""
from __future__ import annotations

import numpy as np

R_BODY = 0.25
TH = dict(near=0.60, response=0.20, still=0.10, moved=0.15, reverse=-0.30, slide=0.50, turn_deg=45.0, head_on_deg=15.0, contact_depth=0.80,
          jump_ratio=3.0, jump_min=2.0, ahead_reach=3.0, centre_deg=9.2, centre_bins=8, centre_fill=0.5, window_vdeg=15.7)
TH2 = dict(inside_depth=0.10, inside_frames=2, behind_ratio=1.1, behind_margin=0.3, jump_inside=0.05, graze=R_BODY - 0.10,
           closeup_depth=0.30, closeup_fill=0.80, closeup_frames=3)
TH3 = dict(retreat=0.30, retreat_s=0.10, past=1.50, past_s=0.375, yaw_deg=45.0, moving=0.30, moving_s=2.0, window=4, back=0.10, off_deg=45.0,
           off_moved=0.10)
NOISE_CAP = 0.30            # m: the noise floor's upper bound (the evaluator's position error on Unreal Engine renders)
RULE, CONTROL_RULE = "nopebench", "nopebench_open_route"
VOLUME_ABOVE, VOLUME_BELOW = 0.2, 0.5      # the collision volume's extent over and under the eye (m)
EPS_END = 0.30              # m, endpoint progress on an open route
REF_MIN_ALONG = 0.15        # m, wrong_direction on an open route: the reference point compared with is at least this far along it


def volume_points(points, *, above=VOLUME_ABOVE, below=VOLUME_BELOW, max_points=4000, seed=0):
    """Points the camera's collision volume can hit (from `below` under the eye to `above` over it), at most max_points."""
    y = points[:, 1]                                        # down from the eyes
    keep = (y > -above) & (y < below)
    pts = points[keep]
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(seed).choice(len(pts), max_points, replace=False)]
    return pts


def band_points(points, *, reach=10.0, max_points=20000, seed=0):
    """Every point in the collision band within `reach` (horizontal) of the camera: all geometry the hand-off frame shows at head height."""
    pts = points[np.hypot(points[:, 0], points[:, 2]) < reach]
    return volume_points(pts, max_points=max_points, seed=seed)


def horizontal_distance(p, pts):
    """[T] distance (x, z) from camera positions p [T, 3] to the nearest of pts [N, 3]."""
    if len(pts) == 0:
        return np.full(len(p), np.inf)
    d = p[:, None, [0, 2]] - pts[None, :, [0, 2]]
    return np.sqrt((d ** 2).sum(-1)).min(1)


def obstacle_ahead(p, u, pts, reach):
    """[T] whether, seen from p, points closer than `reach` fill at least half of the angular bins of the depth window's centre."""
    if len(pts) == 0:
        return np.zeros(len(p), bool)
    u2 = np.array([u[0], u[2]]) / max(np.hypot(u[0], u[2]), 1e-9)
    rel = pts[None, :, [0, 2]] - p[:, None, [0, 2]]
    along = rel @ u2
    side = rel @ np.array([-u2[1], u2[0]])
    ang = np.degrees(np.arctan2(side, np.maximum(along, 1e-6)))
    elev = np.degrees(np.arctan2(pts[None, :, 1] - p[:, None, 1], np.maximum(along, 1e-6)))     # y is down
    ok = (along > 0) & (along < reach[:, None]) & (np.abs(ang) < TH["centre_deg"]) & (np.abs(elev) < TH["window_vdeg"])
    edges = np.linspace(-TH["centre_deg"], TH["centre_deg"], TH["centre_bins"] + 1)
    filled = np.array([[np.any(ok[t] & (ang[t] >= a) & (ang[t] < b)) for a, b in zip(edges[:-1], edges[1:])]
                       for t in range(len(p))])
    return filled.mean(1) >= TH["centre_fill"]


def longest_run(mask):
    """(length, first index) of the longest run of True."""
    best, start, cur = 0, None, 0
    for i, m in enumerate(mask):
        cur = cur + 1 if m else 0
        if cur > best:
            best, start = cur, i - cur + 1
    return best, start


def map_ahead(scene, p, u, reach, step=0.02):
    """[T] rays from p[t] through the window centre's bins (around u[t]) meet eye-level geometry within `reach` in at least half the bins."""
    U = np.broadcast_to(np.asarray(u, float), (len(p), 3))
    edges = np.linspace(-TH["centre_deg"], TH["centre_deg"], TH["centre_bins"] + 1)
    ang = np.radians((edges[:-1] + edges[1:]) / 2)
    r = np.arange(step, max(float(np.max(reach)), step) + step, step)
    out = np.zeros(len(p), bool)
    for t in range(len(p)):
        u2 = np.array([U[t, 0], U[t, 2]]) / max(np.hypot(U[t, 0], U[t, 2]), 1e-9)
        dirs = np.stack([u2[0] * np.cos(ang) + u2[1] * np.sin(ang), -u2[0] * np.sin(ang) + u2[1] * np.cos(ang)], 1)
        rr = r[r <= reach[t]]
        q = p[t, [0, 2]][None, None] + rr[None, :, None] * dirs[:, None, :]           # [bins, steps, (x, z)]
        q3 = np.stack([q[..., 0], np.full(q.shape[:2], p[t, 1]), q[..., 1]], -1).reshape(-1, 3)
        hit = scene.eye_occupied(q3).reshape(len(dirs), len(rr))
        out[t] = hit.any(1).mean() >= TH["centre_fill"]
    return out


def median3(x):
    """Median over three neighbouring samples, the ends as they are."""
    x = np.asarray(x, float)
    return x if len(x) < 3 else np.r_[x[0], np.median(np.stack([x[:-2], x[1:-1], x[2:]]), 0), x[-1]]


def follow_signals(t, p, a, near, n, u_cmd, sig):
    """What the camera does relative to the command on the trusted frames: retreat, going back, distance past the wall, heading, last motion."""
    a0 = np.r_[0.0, a]
    smooth = median3(a0)
    p0 = np.vstack([np.zeros(3), p])[:, [0, 2]]
    cmd = sig.get("cmd_p")
    cmd0 = None if cmd is None else np.vstack([np.zeros(3), cmd])[:, [0, 2]]
    kc = int(np.argmax(near)) if near.any() else len(t)
    win, back, offs = TH3["window"], [], []
    u2 = np.asarray(u_cmd, float)[[0, 2]]
    for i in range(len(p0) - win):
        d = p0[i + win] - p0[i]
        c = u2 if cmd0 is None else cmd0[i + win] - cmd0[i]
        c = c / np.linalg.norm(c) if np.linalg.norm(c) > 1e-6 else u2
        back.append(float(d @ c) < -TH3["back"])
        if i + win <= kc + 1 and np.linalg.norm(d) >= TH3["off_moved"]:
            offs.append(float(np.degrees(np.arccos(np.clip(d @ c / np.linalg.norm(d), -1, 1)))))
    i_last = int(np.argmax(t >= t[-1] - TH3["moving_s"]))
    out = dict(retreat_m=round(float(smooth.max() - a0[-3:].max()), 3), drawdown_m=round(float((np.maximum.accumulate(smooth) - smooth).max()), 3),
               against_frac=round(float(np.mean(back)), 3) if back else 0.0, moved_last_m=round(float(np.linalg.norm((p[-1] - p[i_last])[[0, 2]])), 3),
               pre_contact_off_deg=round(max(offs, default=0.0), 1), pre_contact_off_windows=sum(o > TH3["off_deg"] for o in offs), pre_contact_windows=len(offs),
               past_m=None, yaw_end_deg=None, yaw_max_deg=None)
    if sig.get("reference") is not None:
        out["past_m"] = round(float(median3(-(p @ n)).max() - (-(np.asarray(sig["reference"], float) @ n)).max()), 3)
    if sig.get("yaw_err") is not None:
        yaw = np.asarray(sig["yaw_err"], float)[:len(t)]
        out.update(yaw_end_deg=round(float(abs(np.median(yaw[-3:]))), 1), yaw_max_deg=round(float(np.abs(median3(yaw)).max()), 1))
    return out


def signals(recon, obstacle_pts, u_cmd, scores, handoff, *, area, border, scene, command, reference, pts_all=None, fps=16):
    """Per reconstructed frame after the hand-off (t in s from the hand-off). recon: frames, c2w, depth, intrinsics; scores, area, border:
    the tracker's per-frame values from the hand-off on; scene: task2.scene.SceneField; command [n, 4, 4] and reference [n, 3]: commanded
    poses and reference positions in the hand-off frame from the hand-off on; pts_all: band_points of the hand-off depth (None on an open route)."""
    gen = np.where(recon.frames > handoff)[0]
    p = recon.c2w[gen, :3, 3]
    t = (recon.frames[gen] - handoff) / fps
    s = horizontal_distance(p, obstacle_pts)
    a = p @ u_cmd
    k = recon.frames[gen] - handoff
    present = np.array([scores[i] > 0 for i in k])
    area = np.asarray(area)[k]
    border = np.asarray(border)[k].astype(bool)
    D = np.full(len(gen), np.nan)
    near_frac = np.full(len(gen), np.nan)
    if getattr(recon, "depth", None) is not None:
        for j, g in enumerate(gen):
            d, K = np.asarray(recon.depth[g], float), recon.intrinsics[g]
            hh, ww = d.shape
            col = K[0, 2] + K[0, 0] * u_cmd[0] / max(u_cmd[2], 1e-3)       # where the commanded direction points
            col = float(np.clip(col, ww / 6, ww - ww / 6))
            win = d[hh // 4:3 * hh // 4, int(col - ww / 6):int(col + ww / 6)]
            D[j] = np.percentile(win[np.isfinite(win) & (win > 0)], 20)
            ok = np.isfinite(d) & (d > 0)
            near_frac[j] = float((d[ok] < TH2["closeup_depth"]).mean()) if ok.any() else np.nan
    reach = np.where(np.isfinite(D), np.minimum(D + 0.5, TH["ahead_reach"]), TH["ahead_reach"])
    blocked = obstacle_ahead(p, u_cmd, obstacle_pts, reach)
    # at frame j, judged from where smooth motion would have put the camera (pose j itself breaks at a violation)
    prev = np.vstack([np.zeros((2, 3)), p])
    p_pred = 2 * prev[1:-1] - prev[:-2]
    reach_prev = np.r_[TH["ahead_reach"], reach[:-1]]
    s0 = float(horizontal_distance(np.zeros((1, 3)), obstacle_pts)[0])
    sig = dict(t=t, p=p, s=s, s0=s0, a=a, present=present, area=area, border=border, D=D, blocked=blocked, near_frac=near_frac,
               reference=np.asarray(reference, float))
    want = np.asarray(command, float)[np.minimum(k, len(command) - 1)]
    sig["cmd_p"] = want[:, :3, 3]
    fwd = recon.c2w[gen, :3, 2]
    yaw = np.degrees(np.arctan2(fwd[:, 0], fwd[:, 2]) - np.arctan2(want[:, 0, 2], want[:, 2, 2]))
    sig["yaw_err"] = (yaw + 180.0) % 360.0 - 180.0
    # the window of the frame before, turned with that frame's estimated rotation, so a camera that has turned is not judged along u
    wdir = np.tile(np.asarray(u_cmd, float), (len(gen), 1))
    for j, g in enumerate(gen):
        K = recon.intrinsics[g]
        ww = recon.depth[g].shape[1] if getattr(recon, "depth", None) is not None else 2 * K[0, 2]
        col = float(np.clip(K[0, 2] + K[0, 0] * u_cmd[0] / max(u_cmd[2], 1e-3), ww / 6, ww - ww / 6))
        d = recon.c2w[g, :3, :3] @ np.array([(col - K[0, 2]) / K[0, 0], 0.0, 1.0])
        if np.hypot(d[0], d[2]) > 1e-6:
            wdir[j] = np.array([d[0], 0.0, d[2]]) / np.hypot(d[0], d[2])
    wdir_prev = np.vstack([np.asarray(u_cmd, float)[None], wdir[:-1]])
    sig["window_dir_prev"] = wdir_prev
    if pts_all is not None:
        sig["ahead_pred_all"] = np.array([obstacle_ahead(p_pred[j:j + 1], wdir_prev[j], pts_all, reach_prev[j:j + 1])[0]
                                          for j in range(len(gen))], bool)
    travelled = np.cumsum(np.linalg.norm(np.diff(np.vstack([np.zeros((1, 3)), p])[:, [0, 2]], axis=0), axis=1))
    sd = scene.signed_distance(p)
    fp = scene.free_path(p)
    sig.update(sd=sd, sd_map=scene.map_signed_distance(p), free_path=fp, travelled=travelled,
               detour=fp - TH2["behind_ratio"] * travelled - TH2["behind_margin"])
    # the extrapolated position at frame j, with the path travelled up to frame j - 1 plus the extrapolated step
    trav_pred = np.r_[0.0, travelled[:-1]] + np.linalg.norm((p_pred - prev[1:-1])[:, [0, 2]], axis=1)
    sd_pred = scene.signed_distance(p_pred)
    fp_pred = scene.free_path(p_pred)
    sig.update(sd_pred=sd_pred, behind_pred=(sd_pred >= 0) & (fp_pred > TH2["behind_ratio"] * trav_pred + TH2["behind_margin"]),
               ahead_map=map_ahead(scene, p_pred, wdir_prev, reach_prev))
    return sig


def noise_floor(noise_m):
    """The inside test's depth: the video's history fit error clipped to [TH2['inside_depth'], NOISE_CAP]."""
    return max(TH2["inside_depth"], min(float(noise_m or 0.0), NOISE_CAP))


def outcome(sig, wall_normal, u_cmd, beta_deg, speed_cmd, feasible_slide, noise_m=None):
    """Category, violation flag and progress of one video on an obstacle route. wall_normal: horizontal unit normal of the true wall at the
    contact, towards the camera; feasible_slide: how far the reference execution slides along the wall; noise_m: the video's history fit
    error (the noise floor)."""
    D, blocked = sig["D"], sig["blocked"]
    jumps = [j for j in range(1, len(D)) if np.isfinite(D[j - 1]) and D[j] >= max(TH["jump_min"], TH["jump_ratio"] * D[j - 1])]
    ahead = sig["ahead_pred_all"]
    geo = (np.nan_to_num(sig["sd_pred"], nan=1.0) <= -TH2["jump_inside"]) | sig["behind_pred"] | sig["ahead_map"]
    through_j = [j for j in jumps if D[j - 1] <= TH["contact_depth"] and (ahead[j] or geo[j])]
    vanish_j = [j for j in jumps if D[j - 1] > TH["contact_depth"] and ahead[j] and sig["ahead_map"][j]]
    beside_j = [j for j in jumps if j not in through_j and j not in vanish_j]
    bad = sorted(through_j + vanish_j)
    valid = slice(0, bad[0]) if bad else slice(None)        # pose is trusted before a violating jump only
    t, s, a, p = sig["t"][valid], sig["s"][valid], sig["a"][valid], sig["p"][valid]
    if len(t) == 0:
        t, s, a, p = sig["t"][:1], sig["s"][:1], sig["a"][:1], sig["p"][:1]
    first_1s = t <= 1.0 + 1e-6
    a1 = a[first_1s][-1] if first_1s.any() else 0.0
    v_model = max(a1 / 1.0, 1e-6)
    disp = np.linalg.norm(p, axis=1)
    tail = t >= t[-1] - 1.0
    v_end = float(np.linalg.norm(p[tail][-1] - p[tail][0]) / max(t[tail][-1] - t[tail][0], 1e-6)) if tail.sum() > 1 else 0.0
    near = (s <= TH["near"]) | (D[valid][:len(s)] <= TH["near"])
    n = np.asarray(wall_normal, float)
    tangent = u_cmd - (u_cmd @ n) * n
    tangent[1] = 0.0
    t_hat = tangent / max(np.linalg.norm(tangent), 1e-9)
    if near.any():
        kc = int(np.argmax(near))
        slide = float((p[-1] - p[kc]) @ t_hat)
    else:
        slide = 0.0
    feasible = float(feasible_slide)
    vanished = bool(vanish_j)
    blk = blocked[valid][:len(t)]
    for i in range(1, len(t)):          # obstacle lost by the tracker while it is ahead, in view, not at the border
        if sig["present"][i - 1] and not sig["present"][i] and s[i] > TH["near"] and blk[i - 1] \
                and not sig["border"][i - 1] and 0.01 < sig["area"][i - 1] < 0.5:
            if not sig["ahead_map"][i]:      # the map says the camera has passed it
                continue
            vanished = True
    head_on = beta_deg < TH["head_on_deg"]
    info = dict(depth_jumps=[round(float(sig["t"][j]), 3) for j in jumps],
                jumps_through=[round(float(sig["t"][j]), 3) for j in through_j],
                jumps_beside=[round(float(sig["t"][j]), 3) for j in beside_j],
                min_obstacle_distance_m=round(float(s.min()), 3), start_obstacle_distance_m=round(sig["s0"], 3),
                travel_1s_m=round(float(a1), 3), end_speed_mps=round(v_end, 3), slide_m=round(slide, 3),
                feasible_slide_m=round(feasible, 3), head_on=bool(head_on), vanished=bool(vanished))
    sd = sig["sd"][valid][:len(t)]
    floor = noise_floor(noise_m)
    inside = np.nan_to_num(sd, nan=1.0) <= -floor
    behind = (np.nan_to_num(sd, nan=-1.0) >= TH2["inside_depth"]) & (np.nan_to_num(sig["detour"][valid][:len(t)], nan=-1.0) > 0)
    run, run_at = longest_run(inside | behind)
    geom = run >= TH2["inside_frames"]
    shallow = longest_run((np.nan_to_num(sd, nan=1.0) <= -TH2["inside_depth"]) | behind)[0] >= TH2["inside_frames"] and not geom
    nf = sig["near_frac"][valid][:len(t)]
    crun, crun_at = longest_run(np.nan_to_num(nf, nan=0.0) >= TH2["closeup_fill"])
    closeup = crun >= TH2["closeup_frames"]
    min_sd = float(np.nanmin(sd)) if np.isfinite(sd).any() else float("nan")
    steps = np.linalg.norm(np.diff(np.vstack([np.zeros((1, 3)), p])[:, [0, 2]], axis=0), axis=1)
    info.update(rule=RULE, noise_floor_m=round(floor, 3), min_signed_distance_m=round(min_sd, 3),
                min_map_distance_m=round(float(np.nanmin(sig["sd_map"][valid][:len(t)])), 3),
                max_step_m=round(float(steps.max()), 3),     # > 0.5 m in 0.125 s: the pose estimate broke
                inside_frames=int(inside.sum()), behind_frames=int(behind.sum()), inside_run=int(run),
                inside_t=None if not geom else round(float(t[run_at]), 3),
                closeup_run=int(crun), closeup=bool(closeup),
                closeup_t=None if not closeup else round(float(t[crun_at]), 3),
                max_near_fraction=round(float(np.nanmax(nf)), 3) if np.isfinite(nf).any() else None)
    through = geom or bool(through_j)
    follow = follow_signals(t, p, a, near, n, u_cmd, sig)
    info.update(follow)
    past = follow["past_m"] is not None and follow["past_m"] >= max(TH3["past"], TH3["past_s"] * speed_cmd) and np.isfinite(min_sd) and min_sd < 0
    if through or vanished or past:
        cat = "violation_through" if through else "violation_vanished" if vanished else "violation_past"
    elif disp.max() < TH["moved"]:
        cat = "no_response"
    elif a.min() < TH["reverse"] and a[-1] < 0:
        cat = "reversed"
    elif follow["retreat_m"] >= max(TH3["retreat"], TH3["retreat_s"] * speed_cmd):
        cat = "retreated"
    elif a1 < TH["response"] * speed_cmd and not near.any():
        disp_dir = p[-1] / max(np.linalg.norm(p[-1]), 1e-9)
        cat = "wrong_direction" if np.degrees(np.arccos(np.clip(disp_dir @ u_cmd, -1, 1))) > TH["turn_deg"] \
            else "no_response"
    elif not near.any():
        cat = "still_moving" if follow["moved_last_m"] > TH3["moving"] else "early_stop"
    elif follow["yaw_end_deg"] is not None and follow["yaw_end_deg"] > TH3["yaw_deg"]:
        cat = "turned_away"
    elif not head_on and feasible > 0.05 and slide >= TH["slide"] * feasible:
        cat = "slid"
    elif v_end < TH["still"] * speed_cmd or head_on:
        cat = "stopped"
    else:
        cat = "stopped_not_slid"
    violation = cat.startswith("violation")
    info["grazing"] = bool(not violation and np.isfinite(info["min_signed_distance_m"]) and info["min_signed_distance_m"] < TH2["graze"])
    info["uncertain_contact"] = bool(shallow and not violation)
    if head_on:
        progress = 1.0 if near.any() else float(np.clip(1 - (s.min() - TH["near"]) / max(sig["s0"] - TH["near"], 1e-6), 0, 1))
        success = (not violation) and cat in ("stopped", "slid")
    elif feasible <= 0.05:        # nothing to slide along: stopping is all the route allows
        progress = 1.0 if cat == "stopped" else 0.0
        success = (not violation) and cat == "stopped"
    else:
        progress = float(np.clip(slide / feasible, 0, 1))
        success = (not violation) and cat == "slid"
    if cat in ("reversed", "retreated", "turned_away", "no_response", "wrong_direction"):
        progress = 0.0
    return dict(category=cat, violation=bool(violation), progress=round(progress, 3), sis=bool(success),
                sis_soft=round((0.0 if violation else 1.0) * progress, 3), **info)


def path_coordinates(p, ref):
    """(sigma, rho) of positions p [T, 3] against the reference polyline ref [K, 3], horizontal: arc length of the nearest point (negative
    before its start, beyond its length past its end) and the distance to it."""
    q, r = np.asarray(p, float)[:, [0, 2]], np.asarray(ref, float)[:, [0, 2]]
    seg = np.diff(r, axis=0)
    slen = np.linalg.norm(seg, axis=1)
    keep = slen > 1e-6
    r0, seg, slen = r[:-1][keep], seg[keep], slen[keep]
    if not len(seg):
        return np.zeros(len(q)), np.linalg.norm(q - r[0], axis=1)
    start = np.r_[0.0, np.cumsum(slen)[:-1]]
    t = ((q[:, None, :] - r0[None]) * seg[None]).sum(-1) / slen[None] ** 2           # [T, S]
    tc = np.clip(t, 0.0, 1.0)
    tc[:, 0] = np.minimum(t[:, 0], 1.0)                                                 # before the start
    tc[:, -1] = np.maximum(t[:, -1], 0.0) if len(seg) > 1 else t[:, -1]                 # past the end
    near = r0[None] + tc[..., None] * seg[None]
    dist = np.linalg.norm(q[:, None, :] - near, axis=-1)
    k = np.argmin(np.linalg.norm(q[:, None, :] - (r0[None] + np.clip(t, 0, 1)[..., None] * seg[None]), axis=-1), axis=1)
    i = np.arange(len(q))
    return start[k] + tc[i, k] * slen[k], dist[i, k]


def point_at(ref, s):
    """The reference polyline's point (x, z) at arc length s (clipped to the polyline)."""
    r = np.asarray(ref, float)[:, [0, 2]]
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(r, axis=0), axis=1))]
    s = float(np.clip(s, 0.0, cum[-1]))
    return np.array([np.interp(s, cum, r[:, 0]), np.interp(s, cum, r[:, 1])])


def outcome_control(sig, u_cmd, speed_cmd, noise_m=None, eps=EPS_END):
    """Category, violation flag and endpoint progress of one video of an open route: the camera centre in or through the scene geometry is
    a violation (with the same noise floor); the reference is the command walked for the full clip."""
    D = sig["D"]
    jumps = [j for j in range(1, len(D)) if np.isfinite(D[j - 1]) and D[j] >= max(TH["jump_min"], TH["jump_ratio"] * D[j - 1])]
    geo = (np.nan_to_num(sig["sd_pred"], nan=1.0) <= -TH2["jump_inside"]) | sig["behind_pred"] | sig["ahead_map"]
    through_j = [j for j in jumps if D[j - 1] <= TH["contact_depth"] and geo[j]]
    vanish_j = [j for j in jumps if D[j - 1] > TH["contact_depth"] and sig["ahead_map"][j]]
    beside_j = [j for j in jumps if j not in through_j and j not in vanish_j]
    bad = sorted(through_j + vanish_j)
    valid = slice(0, bad[0]) if bad else slice(None)         # pose is trusted before a violating jump only
    t, p = sig["t"][valid], sig["p"][valid]
    if len(t) == 0:
        valid = slice(0, 1)
        t, p = sig["t"][:1], sig["p"][:1]
    sd = sig["sd"][valid][:len(t)]
    floor = noise_floor(noise_m)
    inside = np.nan_to_num(sd, nan=1.0) <= -floor
    behind = (np.nan_to_num(sd, nan=-1.0) >= TH2["inside_depth"]) & (np.nan_to_num(sig["detour"][valid][:len(t)], nan=-1.0) > 0)
    run, run_at = longest_run(inside | behind)
    geom = run >= TH2["inside_frames"]
    shallow = longest_run((np.nan_to_num(sd, nan=1.0) <= -TH2["inside_depth"]) | behind)[0] >= TH2["inside_frames"] and not geom
    nf = sig["near_frac"][valid][:len(t)]
    crun, crun_at = longest_run(np.nan_to_num(nf, nan=0.0) >= TH2["closeup_fill"])
    closeup = crun >= TH2["closeup_frames"]
    min_sd = float(np.nanmin(sd)) if np.isfinite(sd).any() else float("nan")
    steps = np.linalg.norm(np.diff(np.vstack([np.zeros((1, 3)), p])[:, [0, 2]], axis=0), axis=1)
    through, vanished = bool(geom or through_j), bool(vanish_j)
    ref = np.asarray(sig["reference"], float)
    ref_len = float(np.linalg.norm(np.diff(ref[:, [0, 2]], axis=0), axis=1).sum())
    ref_end = ref[-1, [0, 2]]
    d_ref = float(np.linalg.norm(ref_end))
    end = p[-1, [0, 2]]
    d = float(np.linalg.norm(end - ref_end))
    q = float(np.clip(1.0 - max(0.0, d - eps) / max(d_ref - eps, 1e-6), 0.0, 1.0))
    sigma, rho = path_coordinates(p, ref)
    disp = np.linalg.norm(p[:, [0, 2]], axis=1)
    first_1s = t <= 1.0 + 1e-6
    a = p @ np.asarray(u_cmd, float)
    a1 = float(a[first_1s][-1]) if first_1s.any() else 0.0
    tail = t >= t[-1] - 1.0
    v_end = float(np.linalg.norm(p[tail][-1] - p[tail][0]) / max(t[tail][-1] - t[tail][0], 1e-6)) if tail.sum() > 1 else 0.0
    follow = follow_signals(t, p, sigma, np.zeros(len(t), bool), np.zeros(3), u_cmd, dict(sig, reference=None))
    follow.pop("past_m", None)
    wrong_deg = None
    if np.linalg.norm(end) >= TH["moved"]:
        target = point_at(ref, max(float(sigma[-1]), REF_MIN_ALONG))
        wrong_deg = float(np.degrees(np.arccos(np.clip(end @ target / max(np.linalg.norm(end) * np.linalg.norm(target), 1e-9), -1, 1))))
    p0 = np.vstack([np.zeros((1, 3)), p])[:, [0, 2]]
    win = TH3["window"]
    still = [np.linalg.norm(p0[i + win] - p0[i]) < TH["still"] * speed_cmd * win / 8 for i in range(len(p0) - win)]
    if through or vanished:
        cat = "violation_through" if through else "violation_vanished"
    elif disp.max() < TH["moved"]:
        cat = "no_response"
    elif sigma.min() < TH["reverse"] and sigma[-1] < 0:
        cat = "reversed"
    elif follow["retreat_m"] >= max(TH3["retreat"], TH3["retreat_s"] * speed_cmd):
        cat = "retreated"
    elif follow["yaw_end_deg"] is not None and follow["yaw_end_deg"] > TH3["yaw_deg"]:
        cat = "turned_away"
    elif wrong_deg is not None and wrong_deg > TH["turn_deg"]:
        cat = "wrong_direction"
    elif v_end < TH["still"] * speed_cmd and d > eps:
        cat = "early_stop"
    else:
        cat = "followed"
    violation = cat.startswith("violation")
    res = dict(category=cat, violation=violation, progress=round(q, 3), sis=bool(not violation and cat == "followed"),
               sis_soft=round((0.0 if violation else 1.0) * q, 3), kind="control", control_rule=CONTROL_RULE, rule=RULE,
               endpoint_progress=round(q, 3), end_distance_m=round(d, 3), reference_distance_m=round(d_ref, 3), endpoint_eps_m=eps,
               end_m=[round(float(x), 3) for x in end], reference_end_m=[round(float(x), 3) for x in ref_end],
               along_end_m=round(float(sigma[-1]), 3), reference_length_m=round(ref_len, 3), lateral_end_m=round(float(rho[-1]), 3),
               lateral_max_m=round(float(rho.max()), 3), wrong_direction_deg=None if wrong_deg is None else round(wrong_deg, 1),
               stopped_early=cat == "early_stop", still_s=round(float(sum(still)) / 8, 3), travel_m=round(float(steps.sum()), 3),
               travel_1s_m=round(a1, 3), end_speed_mps=round(v_end, 3), max_displacement_m=round(float(disp.max()), 3),
               depth_jumps=[round(float(sig["t"][j]), 3) for j in jumps], jumps_through=[round(float(sig["t"][j]), 3) for j in through_j],
               jumps_vanished=[round(float(sig["t"][j]), 3) for j in vanish_j], jumps_beside=[round(float(sig["t"][j]), 3) for j in beside_j],
               min_signed_distance_m=round(min_sd, 3), min_map_distance_m=round(float(np.nanmin(sig["sd_map"][valid][:len(t)])), 3),
               max_step_m=round(float(steps.max()), 3), inside_frames=int(inside.sum()), behind_frames=int(behind.sum()), inside_run=int(run),
               inside_t=None if not geom else round(float(t[run_at]), 3), closeup_run=int(crun), closeup=bool(closeup),
               closeup_t=None if not closeup else round(float(t[crun_at]), 3),
               max_near_fraction=round(float(np.nanmax(nf)), 3) if np.isfinite(nf).any() else None,
               trusted_frames=int(len(t)), **follow)
    res["grazing"] = bool(not violation and np.isfinite(min_sd) and min_sd < TH2["graze"])
    res.update(noise_floor_m=round(floor, 3), uncertain_contact=bool(shallow and not violation))
    return res
