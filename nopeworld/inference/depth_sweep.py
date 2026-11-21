"""External depth baseline, the geometry: the 34 candidates' labels on a map of DA3's points instead of the scene's voxels.

After each chunk DA3 reconstructs the real history and the last decoded frames (metres through the history's true path). The points of the
collision band (0.5 m under the camera to 0.2 m over it) of the last TAIL frames make an occupancy map on the labels' 5 cm grid around the
last camera, and the labels' own code gives every candidate's swept clearance h, P = Phi((h - m) / s) and the slide intent A
(nopeworld.head.occupancy). The generation side applies the released collision head's decision rule to P and A, stop-only.
"""
from __future__ import annotations

import numpy as np

from ..head.occupancy import PARAMS, ObstacleMap, candidate_labels, phi, slide_intent_fit

R_BODY = PARAMS["r"]
BAND_ABOVE, BAND_BELOW = 0.2, 0.5           # the camera's collision volume over and under the eye (m)
TAIL = 2                                    # decoded frames whose depth is lifted (at 8 fps: the chunk's last 0.25 s)
STRIDE = 2                                  # depth pixels, every second row and column
VOXEL = 0.05                                # the labels' map cell
HALF_M = 8.0                                # the map reaches this far around the camera
FLIGHT_H = 2.0                              # a drone's root over the ground (m)
SELF_MARGIN = 0.10                          # points this close around the body box are the body itself


def points_in(depth, K, c2w, stride=STRIDE):
    """[N, 3] points of a depth map (metres) in the reconstruction's world (the hand-off camera: x right, y down, z forward)."""
    depth = np.asarray(depth, float)
    h, w = depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    d = depth[::stride, ::stride]
    keep = np.isfinite(d) & (d > 0)
    cam = np.stack([(u[keep] - K[0, 2]) / K[0, 0] * d[keep], (v[keep] - K[1, 2]) / K[1, 1] * d[keep], d[keep], np.ones(keep.sum())], 1)
    return (np.asarray(c2w, float) @ cam.T).T[:, :3]


def band(points, camera_y, *, above=BAND_ABOVE, below=BAND_BELOW):
    """The points of the collision band (y is down): from `above` over the camera to `below` under it."""
    p = np.asarray(points, float)
    return p[(p[:, 1] > camera_y - above) & (p[:, 1] < camera_y + below)]


def depth_map(points_xz, centre_xz, *, voxel=VOXEL, half=HALF_M):
    """An ObstacleMap whose obstacles are the cells DA3's points (x z, metres) fall in."""
    return ObstacleMap.from_points(points_xz, centre_xz, voxel=voxel, half=half)


def level_rotation(up):
    """The rotation that turns `up` (a unit vector in the reconstruction's frame) to OpenCV's up (0, -1, 0)."""
    n_up = np.asarray(up, float) / np.linalg.norm(up)
    target = np.array([0.0, -1.0, 0.0])
    v, c = np.cross(n_up, target), float(n_up @ target)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c) if c > -0.999 else np.diag([1.0, -1.0, -1.0])


def leveled(c2w, R_level):
    T = np.eye(4)
    T[:3, :3] = R_level
    return np.einsum("ij,njk->nik", T, np.asarray(c2w, float))


def in_view(disps_cam, K, width):
    """[C] whether each candidate's end (its displacement in the last camera, x right, z forward) is inside the last frame's horizontal view."""
    d = np.asarray(disps_cam, float)
    half = max(K[0, 2], width - K[0, 2]) / K[0, 0]
    return (d[:, 2] > 0.05) & (np.abs(d[:, 0]) <= half * d[:, 2])


def _cloud(depth, intrinsics, c2w, tail, stride):
    n = len(depth)
    if not 1 <= tail <= n:
        raise ValueError(f"tail {tail} of {n} frames")
    return np.concatenate([points_in(depth[i], intrinsics[i], c2w[i], stride) for i in range(n - tail, n)])


def scores(depth, intrinsics, c2w, disps_cam, *, R_level=None, tail=TAIL, stride=STRIDE):
    """First person: the candidates' labels from a reconstruction whose last `tail` frames are the latest decoded ones. depth [N, h, w]
    metres, intrinsics [N, 3, 3] at the depth's size, c2w [N, 4, 4] in the hand-off camera's frame; R_level turns that frame level (None:
    level already); disps_cam [C, 3]: each candidate's displacement over the next chunk in the last frame's camera, KEEP first.
    Returns (h, P, A, slide, in_view) and how it was measured."""
    c2w = leveled(c2w, np.eye(3) if R_level is None else R_level)
    last = c2w[-1]
    pts = band(_cloud(depth, intrinsics, c2w, tail, stride), last[1, 3])
    world = (last[:3, :3] @ np.asarray(disps_cam, float).T).T[:, [0, 2]]        # the displacements on the level ground plane (x z)
    origin = last[[0, 2], 3]
    omap = depth_map(pts[:, [0, 2]], origin)
    h, P, A, slide = candidate_labels(omap, origin, world, world[0])
    seen = in_view(disps_cam, np.asarray(intrinsics[-1], float), np.asarray(depth[-1]).shape[1])
    return dict(h=h, P=P, A=A, slide=slide, in_view=seen), dict(band_points=int(len(pts)), occupied_cells=int(omap.obstacle.sum()), tail=tail)


def body_band(entity, height):
    """The body's height band over the ground (m) for third person."""
    if entity == "drone":
        return FLIGHT_H - height / 2 - 0.1, FLIGHT_H + height / 2 + 0.1
    return (0.45 if entity == "human" else 0.3), max(height, 0.6)


def footprint(root_xz, forward_xz, box):
    """Points of the body box on the level ground plane (front edge at 5 points, the sides at half the front, the root and the rear)."""
    f = np.asarray(forward_xz, float) / np.linalg.norm(forward_xz)
    r = np.array([-f[1], f[0]])
    pts = []
    for a in (box["front"], box["front"] * 0.5, 0.0, -box["rear"]):
        for b in ((-1.0, -0.5, 0.0, 0.5, 1.0) if a == box["front"] else (-1.0, 1.0)):
            pts.append(np.asarray(root_xz, float) + a * f + b * box["half_width"] * r)
    return np.stack(pts)


def turn(v, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def body_scores(depth, intrinsics, c2w, disps_cam, *, R_level, o_cam, body_turn, entity, box, root_up, tail=TAIL, stride=STRIDE):
    """Third person: the body's labels on a map of DA3's points. The root is the last camera times o_cam, its heading the last camera's
    turned by body_turn; the points of the body's band over its ground, without the body's own (inside its box grown by SELF_MARGIN);
    h is the box footprint swept along each candidate, A against the slide of a disc of half the box's width."""
    c2w = leveled(c2w, R_level)
    last = c2w[-1]
    root = last[:3, :3] @ np.asarray(o_cam, float) + last[:3, 3]
    cam_fwd = last[:3, 2][[0, 2]]
    fwd = turn(cam_fwd / np.linalg.norm(cam_fwd), body_turn)
    ground_y = root[1] + (FLIGHT_H if entity == "drone" else root_up)
    lo, hi = body_band(entity, box["height"])
    pts = _cloud(depth, intrinsics, c2w, tail, stride)
    pts = pts[(pts[:, 1] > ground_y - hi) & (pts[:, 1] < ground_y - lo)]
    rel = pts[:, [0, 2]] - root[[0, 2]]
    a, b = rel @ fwd, rel @ np.array([-fwd[1], fwd[0]])
    own = (a > -box["rear"] - SELF_MARGIN) & (a < box["front"] + SELF_MARGIN) & (np.abs(b) < box["half_width"] + SELF_MARGIN)
    pts = pts[~own]
    world = (last[:3, :3] @ np.asarray(disps_cam, float).T).T[:, [0, 2]]
    omap = depth_map(pts[:, [0, 2]], root[[0, 2]])
    h = np.min([omap.clearance_along(q, world, 0.0) for q in footprint(root[[0, 2]], fwd, box)], axis=0)
    P = phi((h - PARAMS["m"]) / PARAMS["s"])
    d_u = world[0]
    slide = omap.collide_and_slide(root[[0, 2]], d_u, box["half_width"], PARAMS["m"])[0] if np.linalg.norm(d_u) > 1e-6 else np.zeros(2)
    A = slide_intent_fit(world, slide, d_u) if np.linalg.norm(d_u) > 1e-6 else np.ones(len(world))
    ends = np.asarray(o_cam, float)[None] + np.asarray(disps_cam, float)            # the body's end in the last camera
    seen = in_view(ends, np.asarray(intrinsics[-1], float), np.asarray(depth[-1]).shape[1])
    return dict(h=h, P=P, A=A, slide=slide, in_view=seen), dict(band_points=int(len(pts)), own_points=int(own.sum()), occupied_cells=int(omap.obstacle.sum()),
                                                                tail=tail, nearest_to_box_m=round(float(np.min(omap.sdf_at(footprint(root[[0, 2]], fwd, box)))), 3))
