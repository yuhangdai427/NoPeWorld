"""Occupancy maps of the Unreal Engine scenes and the candidate labels computed on them: swept clearance h, P(safe) and the slide intent A.

Unreal world axes, metres: x forward, y right, z up. A map is a top-down slice of the voxels between two heights over the local ground,
turned into a signed distance field (positive in free space). Clearances are measured from the rim of the body disc (radius r):
    P(c) = Phi((h_c - m) / s)                       h_c: rim clearance of the disc swept along candidate c's displacement
    A(c) = exp(-e^2 / (2 sigma^2)),  e^2 = (((d_c - s) . t)^2 + w ((d_c - s) . n)^2) / |d_u|^2
with s the collide-and-slide displacement of the command d_u (walk to the margin, keep the motion along the contact), t its direction
(the command's when s ~ 0), n normal to t; r = 0.25 m, m = 0.05 m, s = 0.03 m, w = 0.5, sigma = 0.5.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from math import erf, sqrt
from pathlib import Path

import numpy as np

PARAMS = dict(r=0.25, m=0.05, s=0.03, sigma_a=0.5)
W_LATERAL = 0.5
CAMERA_BAND = (0.5, 0.2)            # the first-person collision volume: from this far below the eye to this far above it (m)
GEOMETRY_FILE = "geometry_gpu.json"


def phi(z):
    z = np.asarray(z, float)
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))


def rotate(u, deg):
    """Rotate a world xy vector by `deg` in Unreal's yaw sense (+ = clockwise from above = right)."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    u = np.asarray(u, float)
    return np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]])


@dataclass
class MapMeta:
    origin: tuple          # (x0, y0) metres of cell [0, 0]'s lower corner
    voxel: float           # cell size [m]
    floor_z: float         # reference floor height [m]
    step: float            # obstacles start this far above the local ground [m]
    body: float            # ...and end this far above it [m]


class ObstacleMap:
    """2-D obstacle map of a height band over the local ground and its swept-clearance queries."""

    def __init__(self, occ3d: np.ndarray, region_min, voxel: float, floor_z: float, *, step: float = 0.30, body: float = 1.80,
                 ground_window: float = 0.45, ground_ext: np.ndarray | None = None, passable_ext: np.ndarray | None = None):
        """ground_ext: per-cell ground height from physics raycasts (NaN = no walkable ground); passable_ext: occupied cells the engine's
        pawn sweeps straight through (foliage, decals)."""
        from scipy import ndimage
        occ = occ3d.astype(bool)
        nx, ny, nz = occ.shape
        z_centres = region_min[2] + (np.arange(nz) + 0.5) * voxel
        # local ground: the highest occupied voxel at most `ground_window` above the reference floor
        low = z_centres <= floor_z + ground_window
        ground = np.full((nx, ny), floor_z)
        has_ground = np.zeros((nx, ny), bool)
        if ground_ext is not None:
            has_ground = np.isfinite(ground_ext)
            ground = np.where(has_ground, ground_ext, floor_z)
        elif low.any():
            sub = occ[:, :, low]
            has_ground = sub.any(axis=2)
            top = low.nonzero()[0][::-1]
            first = np.argmax(sub[:, :, ::-1], axis=2)          # index from the top of the low band
            ground = np.where(has_ground, z_centres[top[first]], floor_z)
        obstacle = np.zeros((nx, ny), bool)
        for k in range(nz):                 # one z-slice at a time: a full (nx, ny, nz) float array is large
            rel = z_centres[k] - ground
            obstacle |= occ[:, :, k] & (rel > step) & (rel <= body)
        if passable_ext is not None:
            obstacle &= ~passable_ext
        self.solid = obstacle.copy()
        self.obstacle = obstacle
        self.ground = ground
        self.has_ground = has_ground
        self.meta = MapMeta((float(region_min[0]), float(region_min[1])), float(voxel), float(floor_z), float(step), float(body))
        free = ~self.obstacle
        # distances between cell centres; the obstacle surface is half a cell closer
        pos = ndimage.distance_transform_edt(free) * voxel - voxel / 2
        neg = ndimage.distance_transform_edt(self.obstacle) * voxel - voxel / 2
        self.sdf = np.where(free, pos, -neg).astype(np.float32)

    @classmethod
    def from_points(cls, points_xz, centre_xz, *, voxel=0.05, half=8.0):
        """A map whose obstacles are the cells that points (x z, metres) fall in, +-half around centre; outside it sdf_at says obstacle."""
        from scipy import ndimage
        n = int(round(2 * half / voxel))
        origin = np.asarray(centre_xz, float) - half
        occ = np.zeros((n, n), bool)
        q = np.floor((np.asarray(points_xz, float) - origin) / voxel).astype(int) if len(points_xz) else np.zeros((0, 2), int)
        q = q[(q >= 0).all(1) & (q < n).all(1)]
        occ[q[:, 0], q[:, 1]] = True
        m = cls.__new__(cls)
        m.obstacle, m.solid = occ, occ.copy()
        m.ground, m.has_ground = np.zeros((n, n)), np.ones((n, n), bool)
        m.meta = MapMeta((float(origin[0]), float(origin[1])), float(voxel), 0.0, 0.0, 0.0)
        if occ.any():
            pos = ndimage.distance_transform_edt(~occ) * voxel - voxel / 2
            neg = ndimage.distance_transform_edt(occ) * voxel - voxel / 2
            m.sdf = np.where(~occ, pos, -neg).astype(np.float32)
        else:
            m.sdf = np.full((n, n), 2 * half, np.float32)
        return m

    def _idx(self, xy):
        xy = np.atleast_2d(xy)
        return ((xy[:, 0] - self.meta.origin[0]) / self.meta.voxel - 0.5,
                (xy[:, 1] - self.meta.origin[1]) / self.meta.voxel - 0.5)

    def sdf_at(self, xy) -> np.ndarray:
        from scipy import ndimage
        i, j = self._idx(xy)
        return ndimage.map_coordinates(self.sdf, [i, j], order=1, mode="constant", cval=-1.0)

    def normal_at(self, xy) -> np.ndarray:
        """Unit vector pointing away from the nearest obstacle (SDF gradient)."""
        xy = np.atleast_2d(xy)
        e = self.meta.voxel
        gx = self.sdf_at(xy + [e, 0]) - self.sdf_at(xy - [e, 0])
        gy = self.sdf_at(xy + [0, e]) - self.sdf_at(xy - [0, e])
        g = np.stack([gx, gy], axis=1)
        return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)

    def clearance_along(self, p0, disps, r: float, spacing: float = 0.02) -> np.ndarray:
        """Min rim clearance of a disc of radius r swept along straight displacements disps [N, 2]."""
        disps = np.atleast_2d(disps)
        lengths = np.linalg.norm(disps, axis=1)
        n = int(np.ceil(lengths.max() / spacing)) + 1 if len(disps) else 1
        t = np.linspace(0.0, 1.0, max(n, 2))
        pts = np.asarray(p0)[None, None, :] + t[None, :, None] * disps[:, None, :]
        vals = self.sdf_at(pts.reshape(-1, 2)).reshape(len(disps), -1)
        return vals.min(axis=1) - r

    def free_length(self, p0, direction, r: float, m: float, lmax: float, spacing: float = 0.02) -> float:
        """Largest travel along a unit direction keeping rim clearance >= m (0 if already violated)."""
        steps = np.arange(0.0, lmax + spacing, spacing)
        pts = np.asarray(p0)[None, :] + steps[:, None] * np.asarray(direction)[None, :]
        ok = self.sdf_at(pts) - r >= m
        if not ok[0]:
            return 0.0
        bad = np.nonzero(~ok)[0]
        return float(lmax if len(bad) == 0 else max(steps[bad[0]] - spacing, 0.0))

    def collide_and_slide(self, p0, d, r: float, m: float = 0.0, iters: int = 4, skin: float = 1e-3):
        """Kinematic engine move: travel to contact, drop the normal component, repeat."""
        pos = np.asarray(p0, float).copy()
        rem = np.asarray(d, float).copy()
        contacts = []
        for it in range(iters):
            L = float(np.linalg.norm(rem))
            if L < 1e-9:
                break
            e = rem / L
            # at the margin the sampled clearance along the wall dips by millimetres: sliding (and starting inside the margin) may use a
            # 2 cm band below m, never deeper
            c0 = float(self.sdf_at(pos)[0]) - r
            m_eff = m if (it == 0 and c0 >= m) else (m - 0.02 if c0 >= m - 0.02 else c0 - 0.002)
            f = self.free_length(pos, e, r, m_eff, L)
            if f >= L - 1e-6:
                pos = pos + rem
                break
            f = max(f - skin, 0.0)
            pos = pos + f * e
            n = self.normal_at(pos)[0]
            rem = (1.0 - f / L) * rem
            vn = float(rem @ n)
            if vn < 0:
                rem = rem - vn * n
            if any(float(rem @ c) < -1e-9 for c in contacts):
                rem = np.zeros(2)          # crease of two walls: stop
            contacts.append(n)
        return pos - np.asarray(p0, float), contacts


def camera_map(scenes_root, scene, region, eye_m, *, below=CAMERA_BAND[0], above=CAMERA_BAND[1]):
    """ObstacleMap of one saved region in the band from `below` under the eye to `above` over it (the camera's collision volume)."""
    root = Path(scenes_root) / scene
    geo = json.loads((root / GEOMETRY_FILE).read_text())
    reg = next(r for r in geo["regions"] if r["id"] == region)
    z = np.load(root / "geometry" / f"{reg['id']}.npz")
    occ = np.unpackbits(z["occ"], axis=-1)[..., :int(z["occ_shape"][-1])].astype(bool)
    extra = {}
    if "ground" in z:
        extra["ground_ext"] = z["ground"].astype(float)
        if "passable" in z:
            extra["passable_ext"] = np.unpackbits(z["passable"], axis=-1)[..., :z["ground"].shape[1]].astype(bool)
    return ObstacleMap(occ, z["region"], float(z["voxel"]), float(z["floor_z"]), ground_window=reg["band"]["window"],
                       step=eye_m - below, body=eye_m + above, **extra)


def slide_intent_fit(disps, slide, d_u, *, w=W_LATERAL, sigma=PARAMS["sigma_a"]):
    """A of displacements disps [N, 2] for the slide displacement s and the command d_u (same frame, metres)."""
    disps, s, d_u = (np.asarray(x, float) for x in (disps, slide, d_u))
    U = float(np.linalg.norm(d_u))
    if U < 1e-6:
        return np.ones(len(disps))
    S = float(np.linalg.norm(s))
    t = s / S if S > 1e-6 else d_u / U
    n = rotate(t, 90.0)
    delta = disps - s
    e2 = ((delta @ t) ** 2 + w * (delta @ n) ** 2) / U ** 2
    return np.exp(-e2 / (2 * sigma ** 2))


def candidate_labels(omap, p_xy, disps_xy, d_u_xy, params=PARAMS):
    """(h, P, A, slide) of candidate displacements disps from p on a map: the swept clearance, P(safe), the slide intent fit, and the
    command's collide-and-slide displacement."""
    r, m, s = params["r"], params["m"], params["s"]
    disps, d_u = np.atleast_2d(np.asarray(disps_xy, float)), np.asarray(d_u_xy, float)
    h = omap.clearance_along(np.asarray(p_xy, float), disps, r)
    slide = omap.collide_and_slide(np.asarray(p_xy, float), d_u, r, m)[0] if np.linalg.norm(d_u) > 1e-6 else np.zeros(2)
    return h, phi((h - m) / s), slide_intent_fit(disps, slide, d_u), slide
