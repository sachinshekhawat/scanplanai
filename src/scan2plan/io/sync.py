"""Estimate the frame offset between a Stray capture's rgb.mp4 and its odometry/depth.

Measured on the sample captures: the video lags the odometry by a time-varying 0-1.5 s
(e.g. single_scan_with_ceiling: +27..+69 frames across the walk). Depth and poses are
consistent with each other (the LiDAR tier never uses the video), but anything projecting
RGB onto LiDAR geometry (damage regions) or evaluating RGB-only tiers against ARKit poses
must use the offset.

Method: SIFT pure-rotation angle between video frames (i, i+15) vs ARKit rotation angle
between (i+o, i+15+o); per segment, the offset o minimising the median |difference|.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def estimate_offsets(cap, step: int = 23, gap: int = 15, n_seg: int = 6, search=(-120, 151, 3)):
    from scan2plan.photo.pipeline import Photo, _rays, _sift_pairs, rotation_ransac

    vc = cv2.VideoCapture(str(cap.root / "rgb.mp4"))
    n = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    f = cap.K_rgb[0, 0] * 640 / 1920
    idx, est = [], []
    for i in range(100, n - 250, step):
        ph = []
        for k in (i, i + gap):
            vc.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, fr = vc.read()
            if not ok:
                break
            ph.append(Photo(Path("x"), cv2.resize(fr, (640, 480)), f))
        if len(ph) < 2:
            continue
        pr = _sift_pairs(ph)
        if (0, 1) not in pr:
            continue
        R, inl = rotation_ransac(_rays(pr[(0, 1)][0], ph[0]), _rays(pr[(0, 1)][1], ph[1]))
        if R is None or inl.sum() < 30:
            continue
        idx.append(i)
        est.append(_angle(R))
    idx, est = np.array(idx), np.array(est)
    knots, offs = [], []
    for seg in np.array_split(np.arange(len(idx)), n_seg):
        if len(seg) < 3:
            continue
        best = None
        for off in range(*search):
            g = np.array([_angle(cap.rotations[min(max(i + off, 0), n - 1)].T @ cap.rotations[min(max(i + gap + off, 0), n - 1)])
                          for i in idx[seg]])
            e = float(np.median(np.abs(g - est[seg])))
            if best is None or e < best[0]:
                best = (e, off)
        knots.append(float(np.median(idx[seg])))
        offs.append(best[1])
    return np.array(knots), np.array(offs)


def video_to_pose_frame(knots, offs, video_frame: np.ndarray) -> np.ndarray:
    """Odometry frame index that matches a given video frame index."""
    if len(knots) == 0:
        return video_frame
    return np.round(video_frame + np.interp(video_frame, knots, offs)).astype(int)
