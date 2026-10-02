"""Align two plans of the same property (different capture sessions => different world frames).

Both plans are already in Manhattan plan frames, so the residual transform is one of four
90-degree rotations plus a translation. We score each rotation by phase correlation of the
structural wall images and keep the best, then refine translation by matching wall-face
offsets (sub-cm). The result is used only for *evaluation* (repeatability, cross-tier),
never fed back into the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from scan2plan.lidar.rooms import RoomSeg


@dataclass
class Align2D:
    k: int  # number of 90 deg CCW rotations applied to B
    t: np.ndarray  # translation (m) applied after rotation
    score: float

    def apply(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(xy)
        R = np.linalg.matrix_power(np.array([[0, -1], [1, 0]]), self.k)
        return xy @ R.T + self.t


def _wall_points(seg: RoomSeg) -> np.ndarray:
    ys, xs = np.nonzero(seg.wall_img)
    return seg.grid.to_m(np.stack([xs, ys], 1) + 0.5)


def _raster(pts: np.ndarray, lo: np.ndarray, shape: tuple[int, int], res: float) -> np.ndarray:
    img = np.zeros(shape, np.float32)
    ij = np.floor((pts - lo) / res).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] < shape[0])
    img[ij[ok, 1], ij[ok, 0]] = 1
    return cv2.GaussianBlur(img, (0, 0), 2)


def align(a: RoomSeg, b: RoomSeg, res: float = 0.04) -> Align2D:
    pa = _wall_points(a)
    pb0 = _wall_points(b)
    best = None
    for k in range(4):
        R = np.linalg.matrix_power(np.array([[0, -1], [1, 0]]), k)
        pb = pb0 @ R.T
        # centre both on their medians, then phase-correlate on a common canvas
        ca, cb = np.median(pa, 0), np.median(pb, 0)
        span = max(np.ptp(pa, 0).max(), np.ptp(pb, 0).max()) + 4
        n = int(np.ceil(span / res))
        n += n % 2
        lo_a, lo_b = ca - span / 2, cb - span / 2
        ia = _raster(pa, lo_a, (n, n), res)
        ib = _raster(pb, lo_b, (n, n), res)
        win = cv2.createHanningWindow((n, n), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(ib, ia, win)
        t = ca - cb - np.array([dx, dy]) * res
        # verify: fraction of B's wall points landing on A's walls
        shifted = pb + t
        ij = a.grid.idx(shifted)
        dil = cv2.dilate(a.wall_img, np.ones((5, 5), np.uint8))
        score = float(dil[ij[1], ij[0]].mean())
        if best is None or score > best.score:
            best = Align2D(k, t, score)
    return _refine(a, b, best)


def _refine(a: RoomSeg, b: RoomSeg, al: Align2D, iters: int = 3) -> Align2D:
    """Sub-cm translation refinement: median offset between matched long wall faces."""
    R = np.linalg.matrix_power(np.array([[0, -1], [1, 0]]), al.k)
    t = al.t.copy()
    for _ in range(iters):
        d = {0: [], 1: []}
        for rb in b.runs:
            if rb.hi - rb.lo < 0.8:
                continue
            # transform run to A frame
            p = np.array([[rb.c, rb.lo], [rb.c, rb.hi]]) if rb.axis == 0 else np.array([[rb.lo, rb.c], [rb.hi, rb.c]])
            q = p @ R.T + t
            ax = 0 if abs(q[0, 0] - q[1, 0]) < 1e-6 else 1
            n_b = np.zeros(2)
            n_b[rb.axis] = rb.sign
            n_b = R @ n_b
            sign = int(np.sign(n_b[ax]))
            c = q[0, ax]
            lo, hi = sorted(q[:, 1 - ax])
            cands = [ra for ra in a.runs if ra.axis == ax and ra.sign == sign and abs(ra.c - c) < 0.08
                     and min(hi, ra.hi) - max(lo, ra.lo) > 0.5]
            if cands:
                ra = min(cands, key=lambda r: abs(r.c - c))
                d[ax].append(ra.c - c)
        for ax in (0, 1):
            if len(d[ax]) >= 2:
                t[ax] += float(np.median(d[ax]))
    return Align2D(al.k, t, al.score)
