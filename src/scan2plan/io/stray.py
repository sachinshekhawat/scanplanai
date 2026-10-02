"""Loader for Stray Scanner captures (the stock LiDAR capture route).

Layout of one capture directory:
    rgb.mp4               1920x1440 video, one frame per odometry row
    depth/NNNNNN.png      uint16 depth in millimetres, 256x192
    confidence/NNNNNN.png uint8 ARKit confidence {0 low, 1 medium, 2 high}
    odometry.csv          timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, ...
    camera_matrix.csv     3x3 intrinsics at RGB resolution
    imu.csv               timestamp, accel xyz (g), gyro xyz (rad/s)

Conventions (verified empirically: OpenCV convention gives a 17% floor peak in a
2 cm bin, the ARKit -z-forward convention gives 4%):
    camera frame: OpenCV (x right, y down, z forward)
    world frame : ARKit gravity-aligned, +y up, so floor height = min y peak
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

RGB_W, RGB_H = 1920, 1440
DEPTH_W, DEPTH_H = 256, 192


@dataclass
class StrayCapture:
    root: Path
    timestamps: np.ndarray  # (N,)
    frame_ids: np.ndarray  # (N,) int
    positions: np.ndarray  # (N,3) camera centre in world
    rotations: np.ndarray  # (N,3,3) camera->world
    K_rgb: np.ndarray  # (3,3) intrinsics at RGB resolution

    @property
    def n_frames(self) -> int:
        return len(self.frame_ids)

    @property
    def K_depth(self) -> np.ndarray:
        s = DEPTH_W / RGB_W
        K = self.K_rgb.copy()
        K[:2] *= s
        return K

    def pose(self, i: int) -> np.ndarray:
        """4x4 camera->world transform for frame index i."""
        T = np.eye(4)
        T[:3, :3] = self.rotations[i]
        T[:3, 3] = self.positions[i]
        return T

    def depth(self, i: int) -> np.ndarray:
        """Depth in metres (float32, DEPTH_H x DEPTH_W)."""
        d = cv2.imread(str(self.root / "depth" / f"{self.frame_ids[i]:06d}.png"), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(f"missing depth frame {self.frame_ids[i]}")
        return d.astype(np.float32) / 1000.0

    def confidence(self, i: int) -> np.ndarray:
        c = cv2.imread(str(self.root / "confidence" / f"{self.frame_ids[i]:06d}.png"), cv2.IMREAD_UNCHANGED)
        if c is None:
            return np.full((DEPTH_H, DEPTH_W), 2, np.uint8)
        return c

    def backproject(self, i: int, min_conf: int = 2, zmin: float = 0.2, zmax: float = 4.0,
                    stride: int = 1) -> np.ndarray:
        """World-frame points (M,3) from frame i's depth map."""
        d = self.depth(i)[::stride, ::stride]
        c = self.confidence(i)[::stride, ::stride]
        K = self.K_depth
        vv, uu = np.mgrid[0:DEPTH_H:stride, 0:DEPTH_W:stride]
        m = (c >= min_conf) & (d > zmin) & (d < zmax)
        z = d[m]
        pc = np.stack([(uu[m] - K[0, 2]) / K[0, 0] * z, (vv[m] - K[1, 2]) / K[1, 1] * z, z], axis=1)
        return pc @ self.rotations[i].T + self.positions[i]


def find_capture_dir(path: str | Path) -> Path:
    """Accept either the scan dir itself or a parent holding exactly one scan dir."""
    p = Path(path)
    if (p / "odometry.csv").exists():
        return p
    hits = sorted(q.parent for q in p.rglob("odometry.csv"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one Stray capture under {p}, found {len(hits)}")
    return hits[0]


def load_stray(path: str | Path) -> StrayCapture:
    root = find_capture_dir(path)
    od = np.genfromtxt(root / "odometry.csv", delimiter=",", skip_header=1)
    K = np.loadtxt(root / "camera_matrix.csv", delimiter=",")
    quats = od[:, 5:9]  # qx qy qz qw (scipy order)
    return StrayCapture(
        root=root,
        timestamps=od[:, 0],
        frame_ids=od[:, 1].astype(int),
        positions=od[:, 2:5].copy(),
        rotations=Rotation.from_quat(quats).as_matrix(),
        K_rgb=K,
    )
