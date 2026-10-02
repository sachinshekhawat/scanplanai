"""Floor / ceiling level estimation and Manhattan (dominant wall) orientation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Level:
    floor_y: float
    floor_sigma: float  # robust std of floor inliers (m)
    ceiling_y: float | None
    ceiling_sigma: float | None
    ceiling_support: int  # number of ceiling voxels


def _refine_peak(y: np.ndarray, centre: float, half: float = 0.03) -> tuple[float, float, int]:
    sel = y[np.abs(y - centre) < half]
    if len(sel) < 20:
        return centre, half, len(sel)
    med = float(np.median(sel))
    sel = y[np.abs(y - med) < half]
    med = float(np.median(sel))
    mad = float(np.median(np.abs(sel - med)) * 1.4826)
    return med, mad, len(sel)


def estimate_levels(horiz: np.ndarray, cam_y: np.ndarray, bin_m: float = 0.01) -> Level:
    """Floor: densest horizontal slab below the camera. Ceiling: densest slab above the
    highest camera position (phones are held below ceiling height), if well supported."""
    y = horiz[:, 1]
    cam_lo, cam_hi = float(np.percentile(cam_y, 2)), float(np.percentile(cam_y, 98))
    below = y[y < cam_lo - 0.5]
    hist, edges = np.histogram(below, bins=np.arange(below.min(), below.max() + bin_m, bin_m))
    floor_c = edges[np.argmax(hist)] + bin_m / 2
    floor_y, floor_s, _ = _refine_peak(y, floor_c)

    above = y[y > cam_hi + 0.15]
    ceil_y = ceil_s = None
    support = 0
    if len(above) > 500:
        hist, edges = np.histogram(above, bins=np.arange(above.min(), above.max() + bin_m, bin_m))
        c = edges[np.argmax(hist)] + bin_m / 2
        cy, cs, n = _refine_peak(y, c)
        # need enough area: >= ~1.5 m^2 of 1.5cm voxels and a plausible room height
        if n > 1.5 / 0.015**2 * 0.5 and 1.9 < cy - floor_y < 5.0:
            ceil_y, ceil_s, support = cy, cs, n
    return Level(floor_y, floor_s, ceil_y, ceil_s, support)


def dominant_angle(normals_xz: np.ndarray) -> float:
    """Manhattan yaw (rad, in [0, pi/2)) from horizontal wall normals.
    Uses the 4-fold symmetric circular mean: angles*4 averaged on the unit circle,
    refined around the histogram peak to reject non-Manhattan clutter."""
    ang = np.arctan2(normals_xz[:, 1], normals_xz[:, 0])
    a4 = np.mod(ang * 4, 2 * np.pi)
    hist, edges = np.histogram(a4, bins=360, range=(0, 2 * np.pi))
    hist = np.convolve(np.r_[hist[-5:], hist, hist[:5]], np.ones(11) / 11, mode="valid")
    peak = edges[np.argmax(hist)] + np.pi / 360
    d = np.angle(np.exp(1j * (a4 - peak)))
    sel = np.abs(d) < np.deg2rad(5 * 4)
    mean = peak + np.angle(np.mean(np.exp(1j * d[sel])))
    return float(np.mod(mean / 4, np.pi / 2))


def rot2(theta: float) -> np.ndarray:
    """2D rotation that maps the dominant wall direction onto the x axis."""
    c, s = np.cos(-theta), np.sin(-theta)
    return np.array([[c, -s], [s, c]])
