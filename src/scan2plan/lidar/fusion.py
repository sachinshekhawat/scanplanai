"""Gravity-aligned fusion of LiDAR depth frames into classified point sets.

Per frame we back-project high-confidence depth, estimate normals from the depth
image (cross product of neighbour differences, cheap and edge-aware), and split
points into:
    horizontal: |n_y| > cos(15 deg)  -> floor / ceiling / furniture tops
    vertical  : |n_y| < sin(15 deg)  -> walls, doors, cabinet fronts
Everything is voxel-averaged to keep memory bounded on 10k-frame captures.

World frame is ARKit's gravity-aligned frame (+y up). Poses can be overridden
(drift correction supplies corrected poses).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scan2plan.io.stray import DEPTH_H, DEPTH_W, StrayCapture

COS15 = np.cos(np.deg2rad(15))
SIN15 = np.sin(np.deg2rad(15))
N_BEARING = 720  # 0.5 deg bins for plan-view ray carving


@dataclass
class FusedCloud:
    horiz: np.ndarray  # (N,3) world points on horizontal surfaces
    vert: np.ndarray  # (M,3) world points on vertical surfaces
    vert_n: np.ndarray  # (M,2) unit horizontal normal (x,z) of vertical points
    vert_frame: np.ndarray  # (M,) last keyframe index that observed the voxel
    keyframes: np.ndarray  # frame indices used
    fan_range: np.ndarray  # (K, N_BEARING) plan-view free range per world bearing bin (nan = unseen)
    cam_xz: np.ndarray  # (K,2) camera positions of keyframes
    cam_y: np.ndarray  # (K,) camera heights
    meta: dict = field(default_factory=dict)


def select_keyframes(cap: StrayCapture, min_trans: float = 0.05, min_rot_deg: float = 4.0,
                     max_gap: int = 15) -> np.ndarray:
    """Pose-based keyframes: new frame when the camera moved or turned enough."""
    keep = [0]
    last = 0
    cos_thr = np.cos(np.deg2rad(min_rot_deg))
    for i in range(1, cap.n_frames):
        dt = np.linalg.norm(cap.positions[i] - cap.positions[last])
        # angle between optical axes
        c = float(cap.rotations[i][:, 2] @ cap.rotations[last][:, 2])
        if dt > min_trans or c < cos_thr or i - last >= max_gap:
            keep.append(i)
            last = i
    return np.asarray(keep)


def depth_normals(pc: np.ndarray) -> np.ndarray:
    """Normals (H,W,3) from an organised point image (H,W,3) via central differences."""
    dx = np.zeros_like(pc)
    dy = np.zeros_like(pc)
    dx[:, 1:-1] = pc[:, 2:] - pc[:, :-2]
    dy[1:-1, :] = pc[2:, :] - pc[:-2, :]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=2, keepdims=True)
    return n / np.maximum(norm, 1e-9)


def _voxel_mean(pts: np.ndarray, extra: np.ndarray | None, voxel: float):
    keys = np.floor(pts / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((len(counts), pts.shape[1]))
    np.add.at(out, inv, pts)
    out /= counts[:, None]
    ex = None
    if extra is not None:
        ex = np.zeros((len(counts), extra.shape[1]))
        np.add.at(ex, inv, extra)
        ex /= counts[:, None]
    return out, ex, counts


def fuse(cap: StrayCapture, rotations: np.ndarray | None = None, positions: np.ndarray | None = None,
         keyframes: np.ndarray | None = None, zmax: float = 4.0, voxel: float = 0.015,
         min_conf: int = 2) -> FusedCloud:
    Rs = cap.rotations if rotations is None else rotations
    ts = cap.positions if positions is None else positions
    kf = select_keyframes(cap) if keyframes is None else keyframes
    K = cap.K_depth
    vv, uu = np.mgrid[0:DEPTH_H, 0:DEPTH_W]
    rx = (uu - K[0, 2]) / K[0, 0]
    ry = (vv - K[1, 2]) / K[1, 1]

    H, V, Vn, Vf = [], [], [], []
    fans = np.full((len(kf), N_BEARING), np.nan)
    chunk_h, chunk_v, chunk_vn, chunk_vf = [], [], [], []

    def flush():
        if chunk_h:
            h = np.concatenate(chunk_h)
            H.append(_voxel_mean(h, None, voxel)[0])
            chunk_h.clear()
        if chunk_v:
            v = np.concatenate(chunk_v)
            ex = np.concatenate([np.concatenate(chunk_vn), np.concatenate(chunk_vf)[:, None]], 1)
            vm, exm, _ = _voxel_mean(v, ex, voxel)
            V.append(vm); Vn.append(exm[:, :2]); Vf.append(exm[:, 2])
            chunk_v.clear(); chunk_vn.clear(); chunk_vf.clear()

    for j, i in enumerate(kf):
        d = cap.depth(i)
        c = cap.confidence(i)
        pc = np.stack([rx * d, ry * d, d], axis=2)
        n_cam = depth_normals(pc)
        # reject depth discontinuities (flying pixels): neighbour depth jump > 4% of range
        jump = np.zeros_like(d)
        jump[1:-1, 1:-1] = np.maximum(np.abs(d[1:-1, 2:] - d[1:-1, :-2]), np.abs(d[2:, 1:-1] - d[:-2, 1:-1]))
        m = (c >= min_conf) & (d > 0.2) & (d < zmax) & (jump < 0.04 * d + 0.01)
        m[[0, -1], :] = False
        m[:, [0, -1]] = False
        R = Rs[i]
        pw = pc[m] @ R.T + ts[i]
        nw = n_cam[m] @ R.T
        ny = np.abs(nw[:, 1])
        hm = ny > COS15
        vm = ny < SIN15
        chunk_h.append(pw[hm])
        # plan-view carving: every observed point proves the 2D segment camera->point is free
        dxz = pw[:, [0, 2]] - ts[i][[0, 2]]
        rng = np.hypot(dxz[:, 0], dxz[:, 1])
        b = (np.floor((np.arctan2(dxz[:, 1], dxz[:, 0]) + np.pi) / (2 * np.pi) * N_BEARING).astype(int)) % N_BEARING
        if len(b):
            order = np.lexsort((rng, b))
            bs, rs = b[order], rng[order]
            last = np.r_[np.nonzero(np.diff(bs))[0], len(bs) - 1]
            first = np.r_[0, last[:-1] + 1]
            # 90th percentile range per bin: robust to flying pixels / specular outliers
            q = first + ((last - first) * 0.9).astype(int)
            fans[j, bs[last]] = rs[q]
        nxz = nw[vm][:, [0, 2]]
        nxz /= np.maximum(np.linalg.norm(nxz, axis=1, keepdims=True), 1e-9)
        # orient normals towards the camera so both faces of a thin wall separate
        to_cam = ts[i][[0, 2]] - pw[vm][:, [0, 2]]
        flip = np.sum(nxz * to_cam, axis=1) < 0
        nxz[flip] *= -1
        chunk_v.append(pw[vm]); chunk_vn.append(nxz); chunk_vf.append(np.full(vm.sum(), j, float))
        if (j + 1) % 200 == 0:
            flush()
    flush()

    horiz, _, _ = _voxel_mean(np.concatenate(H), None, voxel)
    vall = np.concatenate(V)
    vex = np.concatenate([np.concatenate(Vn), np.concatenate(Vf)[:, None]], 1)
    vert, vex, _ = _voxel_mean(vall, vex, voxel)
    vn = vex[:, :2]
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)
    return FusedCloud(
        horiz=horiz, vert=vert, vert_n=vn, vert_frame=vex[:, 2].astype(int), keyframes=kf, fan_range=fans,
        cam_xz=ts[kf][:, [0, 2]], cam_y=ts[kf][:, 1],
        meta={"voxel": voxel, "zmax": zmax, "min_conf": min_conf, "n_keyframes": int(len(kf))},
    )
