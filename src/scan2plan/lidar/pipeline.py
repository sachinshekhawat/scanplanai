"""LiDAR-tier pipeline: Stray capture -> plan-frame room segmentation (+ levels, drift stats)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from scan2plan.io.stray import StrayCapture, load_stray
from scan2plan.lidar.drift import DriftResult, correct_drift
from scan2plan.lidar.fusion import FusedCloud, fuse
from scan2plan.lidar.planes import Level, dominant_angle, estimate_levels, rot2
from scan2plan.lidar.rooms import RoomSeg, segment_rooms, wall_band


@dataclass
class LidarResult:
    cap: StrayCapture
    cloud: FusedCloud
    level: Level
    theta: float  # Manhattan yaw (rad); plan = R2 @ world_xz
    R2: np.ndarray
    seg: RoomSeg
    drift: DriftResult | None
    timing: dict = field(default_factory=dict)

    def plan_xy(self, world_xz: np.ndarray) -> np.ndarray:
        return world_xz @ self.R2.T


def run_lidar(path, drift: bool = True, log=print) -> LidarResult:
    T = {}
    t0 = time.time()
    cap = load_stray(path)
    rot, pos = cap.rotations, cap.positions
    dres = None
    if drift:
        # coarse pass: levels from a fast sparse fusion, then drift correction on VIO poses
        from scan2plan.lidar.fusion import select_keyframes
        kf = select_keyframes(cap, min_trans=0.25, min_rot_deg=12, max_gap=60)
        coarse = fuse(cap, keyframes=kf, voxel=0.03)
        lv = estimate_levels(coarse.horiz, coarse.cam_y)
        T["coarse_fusion_s"] = time.time() - t0
        t1 = time.time()
        dres = correct_drift(cap, lv.floor_y, wall_band(lv.floor_y, lv.ceiling_y))
        rot, pos = dres.rotations, dres.positions
        T["drift_s"] = time.time() - t1
        log(f"[drift] {dres.stats['n_submaps']} submaps over {dres.stats['path_m']:.1f} m; "
            f"max yaw corr {dres.stats['max_abs_theta_deg']:.2f} deg, max shift {dres.stats['max_abs_shift_m']*100:.1f} cm")
    t1 = time.time()
    cloud = fuse(cap, rotations=rot, positions=pos)
    T["fusion_s"] = time.time() - t1
    t1 = time.time()
    level = estimate_levels(cloud.horiz, cloud.cam_y)
    theta = dominant_angle(cloud.vert_n)
    R2 = rot2(theta)
    v = cloud.vert
    seg = segment_rooms(v[:, [0, 2]] @ R2.T, cloud.vert_n @ R2.T, v[:, 1], cloud.cam_xz @ R2.T,
                        cloud.fan_range, R2, level.floor_y, level.ceiling_y)
    T["layout_s"] = time.time() - t1
    T["total_s"] = time.time() - t0
    log(f"[lidar] {len(seg.polygons)} rooms, {len(seg.openings)} openings, "
        f"floor {level.floor_y:.3f} ceiling {level.ceiling_y}, {T['total_s']:.1f} s")
    return LidarResult(cap, cloud, level, theta, R2, seg, dres, T)
