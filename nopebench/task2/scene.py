"""True scene geometry for the Task 2 rule: where the camera centre of a test video is relative to the real scene.

The map is the route region's camera-band occupancy map (0.5 m below to 0.2 m above the eye over the local ground; 5 cm cells), the same
map the training labels use; the estimated camera path is placed in it through the true hand-off pose. Inside means outside the free
space the hand-off camera stands in: the cells it cannot reach without crossing an occupied cell (4-connected).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nopeworld.head.occupancy import camera_map

S = np.diag([1.0, -1.0, 1.0])       # OpenCV world of the packages' poses <-> Unreal world (x, y, z up), metres
MIN_FREE_M2 = 20.0                  # a hand-off free space smaller than this is map noise: the map's own signed distance is used
EYE_SLICE = 0.15                    # half-height (m) of the eye-level slice
MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1),
         (2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]
_MAPS, _EYE = {}, {}


def region_maps(scenes_root, scene, region, eye_m):
    """(camera-band map, eye-slice obstacle cells) of one region, one region cached at a time."""
    key = (str(scenes_root), scene, region, float(eye_m))
    if key not in _MAPS:
        _MAPS.clear()
        _EYE.clear()
        _MAPS[key] = camera_map(scenes_root, scene, region, eye_m)
        _EYE[key] = camera_map(scenes_root, scene, region, eye_m, below=EYE_SLICE, above=EYE_SLICE).obstacle
    return _MAPS[key], _EYE[key]


def _shifted(a, di, dj, fill):
    """b[i, j] = a[i + di, j + dj], `fill` outside."""
    n0, n1 = a.shape
    b = np.full(a.shape, fill, dtype=a.dtype)
    b[max(-di, 0):n0 + min(-di, 0), max(-dj, 0):n1 + min(-dj, 0)] = a[max(di, 0):n0 + min(di, 0), max(dj, 0):n1 + min(dj, 0)]
    return b


def geodesic(free, src):
    """Path length in cells from cell src through the cells of `free` (16 neighbours; a diagonal step may not pass between two blocked
    cells, a knight step needs both cells it passes free); inf where unreachable."""
    from scipy import sparse
    from scipy.sparse.csgraph import dijkstra
    idx = -np.ones(free.shape, np.int64)
    idx[free] = np.arange(int(free.sum()))
    rows, cols, wts = [], [], []
    for di, dj in MOVES:
        ok = free & _shifted(free, di, dj, False)
        si, sj = int(np.sign(di)), int(np.sign(dj))
        if abs(di) == 1 and abs(dj) == 1:
            ok &= _shifted(free, di, 0, False) | _shifted(free, 0, dj, False)
        elif abs(di) == 2:
            ok &= _shifted(free, si, 0, False) & _shifted(free, si, sj, False)
        elif abs(dj) == 2:
            ok &= _shifted(free, 0, sj, False) & _shifted(free, si, sj, False)
        rows.append(idx[ok])
        cols.append(_shifted(idx, di, dj, -1)[ok])
        wts.append(np.full(int(ok.sum()), float(np.hypot(di, dj))))
    n = int(free.sum())
    g = sparse.csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    dist = dijkstra(g, directed=True, indices=int(idx[src]))
    out = np.full(free.shape, np.inf)
    out[free] = dist
    return out


class SceneField:
    """Signed distance of camera centres (hand-off camera frame) to the free space of the hand-off camera."""

    def __init__(self, case_json, handoff_c2w, scenes_root):
        from scipy import ndimage
        case = json.loads(Path(case_json).read_text())
        self.omap, self.eye_occ = region_maps(scenes_root, case["scene"], case["region"], case["eye_m"])
        self.c2w = np.asarray(handoff_c2w, float)
        occ = self.omap.obstacle
        v = self.omap.meta.voxel
        i, j = self._cell(self.world_xy(np.zeros((1, 3))))
        if occ[i, j]:                                        # map error at the camera: start from the nearest free cell
            _, (ii, jj) = ndimage.distance_transform_edt(occ, return_indices=True)
            i, j = int(ii[i, j]), int(jj[i, j])
        lab, _ = ndimage.label(~occ)                         # default structure: 4-connected
        free = lab == lab[i, j]
        self.free = free
        self.field = np.where(free, ndimage.distance_transform_edt(free) * v - v / 2,
                              -(ndimage.distance_transform_edt(~free) * v - v / 2)).astype(np.float32)
        self.handoff_free_m2 = float(free.sum() * v * v)
        if self.handoff_free_m2 < MIN_FREE_M2:
            self.free = ~occ
            self.field = self.omap.sdf.astype(np.float32)
        self.start = (i, j)
        self._geo = None

    def _geodesic(self, half_m=8.5):
        """Free-space path length (m) from the hand-off camera within a window of +-half_m; NaN off the window."""
        if self._geo is None:
            v = self.omap.meta.voxel
            h = int(round(half_m / v))
            i, j = self.start
            a0, b0 = max(i - h, 0), max(j - h, 0)
            win = self.free[a0:i + h + 1, b0:j + h + 1]
            g = np.full(self.free.shape, np.nan)
            g[a0:i + h + 1, b0:j + h + 1] = geodesic(win, (i - a0, j - b0)) * v
            self._geo = g
        return self._geo

    def free_path(self, p):
        """[T] free-space path length (m) from the hand-off camera to positions p: inf without a path in the window, NaN off it."""
        i, j = self.omap._idx(self.world_xy(p))
        g = self._geodesic()
        ii, jj = np.round(i).astype(int), np.round(j).astype(int)
        ok = (ii >= 0) & (ii < g.shape[0]) & (jj >= 0) & (jj < g.shape[1])
        out = np.full(len(ii), np.nan)
        out[ok] = g[ii[ok], jj[ok]]
        return out

    def world_xy(self, p):
        """Unreal (x, y) in metres of positions p [T, 3] in the hand-off camera frame."""
        p = np.atleast_2d(np.asarray(p, float))
        X = p @ self.c2w[:3, :3].T + self.c2w[:3, 3]
        return (X @ S.T)[:, :2]

    def _cell(self, xy):
        i, j = self.omap._idx(xy)
        n = self.omap.obstacle.shape
        return int(np.clip(np.round(i[0]), 0, n[0] - 1)), int(np.clip(np.round(j[0]), 0, n[1] - 1))

    def signed_distance(self, p):
        """[T] metres for positions p [T, 3] in the hand-off camera frame; NaN off the map."""
        from scipy import ndimage
        i, j = self.omap._idx(self.world_xy(p))
        return ndimage.map_coordinates(self.field, [i, j], order=1, mode="constant", cval=np.nan)

    def eye_occupied(self, p):
        """[T] whether positions p (hand-off camera frame) fall in an occupied cell of the eye-level slice."""
        i, j = self.omap._idx(self.world_xy(p))
        ii, jj = np.round(i).astype(int), np.round(j).astype(int)
        n = self.eye_occ.shape
        ok = (ii >= 0) & (ii < n[0]) & (jj >= 0) & (jj < n[1])
        out = np.zeros(len(ii), bool)
        out[ok] = self.eye_occ[ii[ok], jj[ok]]
        return out

    def map_signed_distance(self, p):
        """[T] the occupancy map's own signed distance (-1 off the map)."""
        return self.omap.sdf_at(self.world_xy(p))
