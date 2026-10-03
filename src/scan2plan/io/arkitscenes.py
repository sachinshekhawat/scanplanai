"""Loader for Apple ARKitScenes `raw` captures (iPad Pro LiDAR), used as an external benchmark
with laser-scanner ground truth (CC BY-NC-SA 4.0, https://github.com/apple/ARKitScenes).

Layout of one video directory:
    lowres_depth/<vid>_<ts>.png     uint16 depth mm, 256x192 (~60 Hz)
    confidence/<vid>_<ts>.png       uint8 {0,1,2}
    lowres_wide/<vid>_<ts>.png      RGB 256x192
    lowres_wide_intrinsics/*.pincam "w h fx fy cx cy"
    lowres_wide.traj                ts, axis-angle (3), translation (3): world->camera, ~10 Hz

Each trajectory sample is paired with the nearest depth frame (|dt| < 20 ms). The world frame
is rotated so gravity is +y (the ARKitScenes world may be z-up); the up axis is detected as the
world axis most aligned with the cameras' mean image-up direction.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scan2plan.io.stray import StrayCapture


def is_arkitscenes(p: Path) -> bool:
    return (p / "lowres_wide.traj").exists() and (p / "lowres_depth").is_dir()


class ARKitScenesCapture(StrayCapture):
    def __init__(self, root, ts, Rcw, tcw, K, depth_paths, conf_paths, rgb_paths):
        super().__init__(root=root, timestamps=ts, frame_ids=np.arange(len(ts)), positions=tcw,
                         rotations=Rcw, K_rgb=K)
        self._d, self._c, self._rgb = depth_paths, conf_paths, rgb_paths

    @property
    def K_depth(self) -> np.ndarray:  # intrinsics are given at depth resolution already
        return self.K_rgb.copy()

    def depth(self, i):
        return cv2.imread(str(self._d[i]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    def confidence(self, i):
        c = cv2.imread(str(self._c[i]), cv2.IMREAD_UNCHANGED) if self._c[i] else None
        return c if c is not None else np.full((192, 256), 2, np.uint8)

    def rgb(self, i):
        return cv2.imread(str(self._rgb[i])) if self._rgb[i] else None


def _ts(p: Path) -> float:
    return float(p.stem.split("_", 1)[1])


def load_arkitscenes(path) -> ARKitScenesCapture:
    root = Path(path)
    traj = np.loadtxt(root / "lowres_wide.traj")
    depth = sorted((root / "lowres_depth").glob("*.png"))
    dts = np.array([_ts(p) for p in depth])
    conf = {p.name: p for p in (root / "confidence").glob("*.png")}
    rgb = {round(_ts(p), 3): p for p in (root / "lowres_wide").glob("*.png")}
    pin = sorted((root / "lowres_wide_intrinsics").glob("*.pincam"))
    w, h, fx, fy, cx, cy = np.loadtxt(pin[0])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    Rs, ts, dp, cp, rp, tt = [], [], [], [], [], []
    for row in traj:
        j = int(np.argmin(np.abs(dts - row[0])))
        if abs(dts[j] - row[0]) > 0.02:
            continue
        Rwc_inv, _ = cv2.Rodrigues(row[1:4])  # world -> camera
        t = row[4:7]
        R = Rwc_inv.T  # camera -> world
        Rs.append(R)
        ts.append(-R @ t)
        dp.append(depth[j])
        cp.append(conf.get(depth[j].name))
        rp.append(rgb.get(round(dts[j], 3)))
        tt.append(row[0])
    Rs, ts = np.array(Rs), np.array(ts)
    # gravity: the world axis best aligned with the mean camera 'up' (-y in OpenCV camera)
    up_w = -np.mean(Rs[:, :, 1], axis=0)
    ax = int(np.argmax(np.abs(up_w)))
    sgn = np.sign(up_w[ax])
    G = np.eye(3)
    if ax == 2:  # z-up world -> y-up: (x, y, z) -> (x, z, -y) with sign
        G = np.array([[1, 0, 0], [0, 0, sgn], [0, -sgn, 0.0]])
    elif ax == 1 and sgn < 0:
        G = np.diag([1, -1, -1.0])
    elif ax == 0:
        G = np.array([[0, sgn, 0], [-sgn, 0, 0], [0, 0, 1.0]])
    Rs = np.einsum("ij,njk->nik", G, Rs)
    ts = ts @ G.T
    cap = ARKitScenesCapture(root, np.array(tt), Rs, ts, K, dp, cp, rp)
    cap.world_to_ours = G  # maps original ARKitScenes world coords into ours
    return cap
