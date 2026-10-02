"""Drift correction: plane-anchored, Manhattan-regularised submap pose graph with loop closure.

ARKit VIO drifts slowly in yaw and horizontal position over long walks (visible as doubled
walls where a room is revisited, and as non-rigid mismatch between two captures of the same
flat). We correct it without any external reference:

 1. Split keyframes into submaps of ~SEG_LEN metres of walk. Each submap s gets a 2D rigid
    correction (theta_s, tx_s, tz_s) about its own centroid, plus a vertical offset dy_s.
 2. Residuals (Gauss-Newton on the linearised small-angle model, a few outer iterations):
    - loop closure / overlap: point-to-plane distance between wall points of submap s and
      the nearest compatible wall point (same normal) from any *non-adjacent* submap s';
      coarse-to-fine gating 0.30 -> 0.06 m. This is what revisits of a room contribute.
    - Manhattan anchor: each submap's wall normals, after correction, should lie on the
      global dominant axes (fixes yaw drift even without revisits).
    - odometry prior: corrections of consecutive submaps should be similar (VIO is locally
      accurate), weighted by a per-metre drift budget.
    - floor anchor (vertical): each submap's floor level equals the global floor.
 3. Corrections are interpolated per frame (by cumulative path length) and applied to all
    poses, so the full-resolution fusion is re-run on corrected poses.

Ablation: pipeline flag `drift=False` uses VIO poses as-is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from scan2plan.io.stray import DEPTH_H, DEPTH_W, StrayCapture
from scan2plan.lidar.fusion import depth_normals, select_keyframes
from scan2plan.lidar.planes import dominant_angle

SEG_LEN = 2.5  # metres of walk per submap


@dataclass
class DriftResult:
    rotations: np.ndarray  # corrected (N,3,3)
    positions: np.ndarray  # corrected (N,3)
    seg_theta: np.ndarray  # per-submap yaw correction (rad)
    seg_t: np.ndarray  # per-submap xz translation (m)
    seg_dy: np.ndarray
    stats: dict


def _sample(cap: StrayCapture, kf: np.ndarray, stride: int = 4, zmax: float = 3.5):
    """Per-keyframe sparse wall (vertical) and floor-candidate (horizontal) points, camera frame."""
    K = cap.K_depth
    vv, uu = np.mgrid[0:DEPTH_H:stride, 0:DEPTH_W:stride]
    rx = (uu - K[0, 2]) / K[0, 0]
    ry = (vv - K[1, 2]) / K[1, 1]
    out = []
    for i in kf:
        d = cap.depth(i)[::stride, ::stride]
        c = cap.confidence(i)[::stride, ::stride]
        pc = np.stack([rx * d, ry * d, d], 2)
        n = depth_normals(pc)
        m = (c >= 2) & (d > 0.3) & (d < zmax)
        m[[0, -1], :] = False
        m[:, [0, -1]] = False
        R = cap.rotations[i]
        nw = n[m] @ R.T
        out.append((pc[m] @ R.T, nw))  # world-oriented, camera-centred
    return out


def _rot2(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def correct_drift(cap: StrayCapture, floor_y: float, wall_band: tuple[float, float],
                  iters: int = 8, w_odo: float = 30.0, w_manh: float = 4.0,
                  gates=(0.30, 0.20, 0.12, 0.08, 0.06, 0.06, 0.05, 0.05)) -> DriftResult:
    kf = select_keyframes(cap, min_trans=0.10, min_rot_deg=8.0, max_gap=30)
    pos = cap.positions
    path = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pos[:, [0, 2]], axis=0), axis=1))]
    seg_of_kf = np.minimum((path[kf] / SEG_LEN).astype(int), int(path[-1] / SEG_LEN))
    _, seg_of_kf = np.unique(seg_of_kf, return_inverse=True)
    S = seg_of_kf.max() + 1
    samples = _sample(cap, kf)

    # collect wall points (xz, normal xz) and floor points (y) per submap, in world frame
    P, N, SID, FY, FS = [], [], [], [], []
    for j, (pw_c, nw) in enumerate(samples):
        i = kf[j]
        pw = pw_c + pos[i]
        ny = np.abs(nw[:, 1])
        vm = (ny < 0.2) & (pw[:, 1] > wall_band[0]) & (pw[:, 1] < wall_band[1])
        nxz = nw[vm][:, [0, 2]]
        nxz /= np.maximum(np.linalg.norm(nxz, axis=1, keepdims=True), 1e-9)
        P.append(pw[vm][:, [0, 2]]); N.append(nxz); SID.append(np.full(vm.sum(), seg_of_kf[j]))
        fm = (ny > 0.95) & (np.abs(pw[:, 1] - floor_y) < 0.15)
        FY.append(pw[fm, 1]); FS.append(np.full(fm.sum(), seg_of_kf[j]))
    P, N, SID = np.concatenate(P), np.concatenate(N), np.concatenate(SID)
    FY, FS = np.concatenate(FY), np.concatenate(FS)

    # voxel thin per submap (5 cm) so dense submaps do not dominate
    key = np.c_[np.floor(P / 0.05).astype(np.int64), SID]
    _, first = np.unique(key, axis=0, return_index=True)
    P, N, SID = P[first], N[first], SID[first]

    theta0 = dominant_angle(N)
    centres = np.array([P[SID == s].mean(0) if (SID == s).any() else np.zeros(2) for s in range(S)])
    th = np.zeros(S)
    t = np.zeros((S, 2))
    hist = []

    def transformed():
        out = np.empty_like(P)
        nout = np.empty_like(N)
        for s in range(S):
            m = SID == s
            Rs = _rot2(th[s])
            out[m] = (P[m] - centres[s]) @ Rs.T + centres[s] + t[s]
            nout[m] = N[m] @ Rs.T
        return out, nout

    rng = np.random.default_rng(0)
    for it in range(iters):
        gate = gates[min(it, len(gates) - 1)]
        Q, NQ = transformed()
        tree = cKDTree(Q)
        # query a subsample of points; match to nearest point of a non-adjacent submap with same normal
        qi = rng.choice(len(Q), size=min(len(Q), 60000), replace=False)
        dd, nn = tree.query(Q[qi], k=12, distance_upper_bound=gate)
        rows_i, rows_j = [], []
        for col in range(nn.shape[1]):
            ok = np.isfinite(dd[:, col])
            a = qi[ok]
            b = nn[ok, col]
            ok2 = (np.abs(SID[a] - SID[b]) > 1) & (np.sum(NQ[a] * NQ[b], 1) > 0.9)
            rows_i.append(a[ok2]); rows_j.append(b[ok2])
        A_i = np.concatenate(rows_i)
        A_j = np.concatenate(rows_j)
        # one match per query point (the first valid neighbour)
        _, u = np.unique(A_i, return_index=True)
        A_i, A_j = A_i[u], A_j[u]

        nvar = 3 * S
        H = np.zeros((nvar, nvar))
        g = np.zeros(nvar)
        J2 = np.array([[0, -1], [1, 0]])
        if len(A_i):
            n = NQ[A_j]
            r = np.sum(n * (Q[A_i] - Q[A_j]), 1)
            # robust (Huber) weights
            sc = max(np.median(np.abs(r)) * 1.4826, 0.01)
            w = np.minimum(1.0, 1.5 * sc / np.maximum(np.abs(r), 1e-9))
            si, sj = SID[A_i], SID[A_j]
            # d r / d theta_si = n . J (q_i - c_si);  d r / d t_si = n ; and negatives for sj
            dth_i = np.sum(n * ((Q[A_i] - centres[si] - t[si]) @ J2.T), 1)
            dth_j = -np.sum(n * ((Q[A_j] - centres[sj] - t[sj]) @ J2.T), 1)
            jac = np.zeros((len(r), 6))
            jac[:, 0], jac[:, 1:3] = dth_i, n
            jac[:, 3], jac[:, 4:6] = dth_j, -n
            idx = np.stack([3 * si, 3 * si + 1, 3 * si + 2, 3 * sj, 3 * sj + 1, 3 * sj + 2], 1)
            for a_ in range(6):
                np.add.at(g, idx[:, a_], w * jac[:, a_] * r)
                for b_ in range(6):
                    np.add.at(H, (idx[:, a_], idx[:, b_]), w * jac[:, a_] * jac[:, b_])
            hist.append(float(np.median(np.abs(r))))
        # Manhattan anchor per submap: mean 4-fold angle residual of its normals
        ang = np.arctan2(NQ[:, 1], NQ[:, 0])
        d4 = np.angle(np.exp(1j * 4 * (ang - theta0))) / 4
        for s in range(S):
            m = (SID == s) & (np.abs(d4) < np.deg2rad(8))
            if m.sum() < 30:
                continue
            res = np.median(d4[m])
            wm = w_manh * min(m.sum(), 2000)
            H[3 * s, 3 * s] += wm
            g[3 * s] += wm * res
        # odometry prior between consecutive submaps (drift budget ~1 deg, 2 cm per submap)
        for s in range(S - 1):
            for k, scale in ((0, np.deg2rad(1.0)), (1, 0.02), (2, 0.02)):
                wv = w_odo / scale**2 * 1e-3
                a_, b_ = 3 * s + k, 3 * (s + 1) + k
                cur = (th[s] - th[s + 1]) if k == 0 else (t[s, k - 1] - t[s + 1, k - 1])
                H[a_, a_] += wv; H[b_, b_] += wv; H[a_, b_] -= wv; H[b_, a_] -= wv
                g[a_] += wv * cur; g[b_] -= wv * cur
        # gauge: weak prior to zero on all, strong on submap 0
        H += np.eye(nvar) * 1e-2
        H[0:3, 0:3] += np.eye(3) * 1e6
        dx = -np.linalg.solve(H, g)
        th += dx[0::3]
        t[:, 0] += dx[1::3]
        t[:, 1] += dx[2::3]

    # vertical: per submap floor offset, smoothed
    dy = np.zeros(S)
    for s in range(S):
        f = FY[FS == s]
        if len(f) > 50:
            dy[s] = floor_y - np.median(f)
    dy = np.convolve(np.pad(dy, 2, mode="edge"), np.ones(5) / 5, mode="valid")

    # per-frame interpolation by path length between submap centres
    seg_path = np.array([np.median(path[kf][seg_of_kf == s]) for s in range(S)])
    fth = np.interp(path, seg_path, th)
    ftx = np.interp(path, seg_path, t[:, 0])
    ftz = np.interp(path, seg_path, t[:, 1])
    fcx = np.interp(path, seg_path, centres[:, 0])
    fcz = np.interp(path, seg_path, centres[:, 1])
    fdy = np.interp(path, seg_path, dy)
    newR = np.empty_like(cap.rotations)
    newP = pos.copy()
    for i in range(cap.n_frames):
        c, s_ = np.cos(fth[i]), np.sin(fth[i])
        M = np.array([[c, 0, -s_], [0, 1, 0], [s_, 0, c]])
        newR[i] = M @ cap.rotations[i]
        xz = pos[i, [0, 2]] - [fcx[i], fcz[i]]
        xz = _rot2(fth[i]) @ xz + [fcx[i], fcz[i]] + [ftx[i], ftz[i]]
        newP[i, 0], newP[i, 2] = xz
        newP[i, 1] += fdy[i]
    stats = {
        "n_submaps": int(S), "n_keyframes": int(len(kf)), "path_m": float(path[-1]),
        "match_median_residual_m": hist, "max_abs_theta_deg": float(np.rad2deg(np.abs(th).max())),
        "max_abs_shift_m": float(np.linalg.norm(t, axis=1).max()), "max_abs_dy_m": float(np.abs(dy).max()),
    }
    return DriftResult(newR, newP, th, t, dy, stats)
