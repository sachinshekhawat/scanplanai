"""Video tier: handheld RGB walkthrough (any iPhone) -> same Plan contract as LiDAR.

Only the video file is used (no depth, poses, IMU or intrinsics from the device).

 1. Frames: decode at VIDEO_FPS, longest side 1024 px.
 2. Poses + intrinsics: COLMAP SIFT, sequential matching with loop pairs, incremental SfM
    (pycolmap; focal initialised from the iPhone main-camera FOV and refined in BA).
 3. Dense depth: Depth-Anything-V2 Metric-Indoor (Small) per keyframe, aligned per frame
    to the SfM sparse depths (robust median ratio), which makes depth consistent across frames.
 4. Metric scale: the median over frames of (DA depth / SfM depth) gives metres per SfM
    unit; corrected by DA_SCALE_BIAS measured on LiDAR captures (disclosed in report).
 5. Gravity: mean camera "down" axis, refined by the normal of the dominant horizontal
    surface cluster (the floor).
 6. A `PseudoCapture` exposes the StrayCapture interface, so fusion, drift correction,
    room segmentation and openings run unchanged. Error model: video sigmas (scale term).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from scan2plan.io.stray import DEPTH_H, DEPTH_W

VIDEO_FPS = 6.0
MAX_FRAMES = 700  # ~15 min end-to-end on an M1 for a 3-4 min walk
MAX_SIDE = 1024
DA_MODEL = "depth-anything/DA3METRIC-LARGE"
# DA3METRIC depth / LiDAR depth on upright frames with the iPhone focal prior (f = 0.83 x long
# side), scripts/calibrate_metric_depth.py -> benchmark/results/metric_depth_calibration.json:
# per capture 0.934 / 0.940 / 0.854, pooled 0.933, per-frame log-sigma 0.118, capture spread 0.039.
DA_SCALE_BIAS = 0.933
VIDEO_SCALE_REL = 0.04  # capture-level spread of the bias (many frames average the per-frame term)


def find_video(p: Path) -> Path:
    if p.is_file():
        return p
    for pat in ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.m4v"):
        hits = sorted(p.rglob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no video under {p}")


def extract_frames(video: Path, out: Path, fps: float = VIDEO_FPS) -> list[Path]:
    """Decode every frame, keep the sharpest (variance of Laplacian) in each 1/fps window."""
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("*.jpg"))
    key = out / ".decode_params"
    want = f"fps={fps} max_frames={MAX_FRAMES} max_side={MAX_SIDE}"
    if existing and key.exists() and key.read_text() == want:
        return existing
    for p in existing:
        p.unlink()
    key.write_text(want)
    cap = cv2.VideoCapture(str(video))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    # cap runtime on long walks: at most MAX_FRAMES keyframes (fps lowered accordingly)
    fps = min(fps, MAX_FRAMES * vfps / n_total)
    step = max(vfps / fps, 1.0)
    i, win, best = 0, 0, None
    paths = []

    def flush(b):
        f = b[2]
        s = MAX_SIDE / max(f.shape[:2])
        if s < 1:
            f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        p = out / f"{b[1]:06d}.jpg"
        cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(p)

    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (512, 384)), cv2.COLOR_BGR2GRAY)
        sh = float(cv2.Laplacian(g, cv2.CV_32F).var())
        if best is None or sh > best[0]:
            best = (sh, i, f)
        if i + 1 >= (win + 1) * step:
            flush(best)
            best, win = None, win + 1
        i += 1
    if best is not None:
        flush(best)
    return paths


def run_sfm(img_dir: Path, work: Path, log=print) -> list[dict]:
    """COLMAP in a child process (pycolmap and torch ship clashing libomp builds on macOS).
    Returns the list of SfM fragments (each internally consistent, arbitrary similarity frame)."""
    import subprocess
    import sys

    f = work / "sparse" / "export.npz"
    n_frames = len(list(img_dir.glob("*.jpg")))
    stamp = work / "sparse" / "frames.txt"
    stale = (not stamp.exists()) or stamp.read_text().strip() != str(n_frames)
    if stale or not f.exists() or len(np.load(f, allow_pickle=True)["data"]) == 0:
        cmd = [sys.executable, "-m", "scan2plan.video.sfm_worker", str(img_dir), str(work)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        for line in p.stdout.splitlines():
            if line.startswith("[video]"):
                log(line)
        if p.returncode != 0:
            raise RuntimeError(f"SfM failed ({p.returncode}): {p.stderr[-2000:]}")
        stamp.write_text(str(n_frames))
    return list(np.load(f, allow_pickle=True)["data"])


def depth_cache(frames: list[Path], cache: Path, log=print, orient: int = 0) -> dict[str, np.ndarray]:
    """Metric depth per frame (DA3METRIC-LARGE in a child process) on upright frames, rotated
    back to the stored frame orientation; deterministic replayable cache (float16 npz)."""
    import subprocess
    import sys

    f = cache / f"da3m_depth_o{orient}.npz"
    src_frames = frames
    frames = upright_frames(frames, orient, cache / f"frames_o{orient}")
    if f.exists():
        z = np.load(f)
        if all(p.stem in z.files for p in frames):
            log(f"[video] depth cache hit ({len(frames)} frames)")
            return _unrotate_depths({k: z[k].astype(np.float32) for k in z.files}, orient)
    cache.mkdir(parents=True, exist_ok=True)
    lst = cache / "depth_list.txt"
    lst.write_text("\n".join(str(p.resolve()) for p in frames))
    t = time.time()
    p = subprocess.run([sys.executable, "-m", "scan2plan.learned.da3_worker", "metric", str(lst), str(f)],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"depth worker failed: {p.stderr[-2000:]}")
    log(f"[video] DA3-metric depth on {len(frames)} frames in {time.time() - t:.0f} s")
    z = np.load(f)
    return _unrotate_depths({k: z[k].astype(np.float32) for k in z.files}, orient)


def _unrotate_depths(d: dict, orient: int) -> dict:
    if orient == 0:
        return d
    out = {}
    for k, v in d.items():
        # worker resizes to 256x192 regardless of orientation; restore the upright aspect first
        up = cv2.resize(v, (DEPTH_H, DEPTH_W)) if orient in (1, 3) else v
        out[k] = cv2.resize(cv2.rotate(up, UNROT[orient]), (DEPTH_W, DEPTH_H))
    return out


@dataclass
class PseudoCapture:
    """StrayCapture-compatible view of a reconstructed video."""
    rotations: np.ndarray
    positions: np.ndarray
    K_depth: np.ndarray
    depths: list[np.ndarray]
    confs: list[np.ndarray]
    timestamps: np.ndarray
    names: list[str]

    @property
    def n_frames(self) -> int:
        return len(self.depths)

    def depth(self, i):
        return self.depths[i]

    def confidence(self, i):
        return self.confs[i]

    def pose(self, i):
        T = np.eye(4)
        T[:3, :3] = self.rotations[i]
        T[:3, 3] = self.positions[i]
        return T


def _gravity_from_normals(Rs: np.ndarray, depths: list[np.ndarray], K: np.ndarray) -> np.ndarray:
    """World 'up' without assuming phone orientation (frames may be stored rotated 90 deg).

    Candidates are the camera x and y axes (the phone is held upright in portrait or landscape,
    so one of them is vertical) averaged over frames; the vertical one is the candidate most
    consistent with the dominant planar normals in the depth maps and most orthogonal to the
    optical axes. Sign: floor-like surfaces must lie *below* the cameras."""
    from scan2plan.lidar.fusion import depth_normals

    vv, uu = np.mgrid[0:DEPTH_H:4, 0:DEPTH_W:4]
    ns, pts = [], []
    for i in range(0, len(depths), max(len(depths) // 40, 1)):
        d = depths[i][::4, ::4]
        P = np.stack([(uu - K[0, 2]) / K[0, 0] * d, (vv - K[1, 2]) / K[1, 1] * d, d], 2)
        # depth_normals points away from the camera; flip so visible surfaces face the camera
        ns.append(-depth_normals(P)[1:-1, 1:-1].reshape(-1, 3) @ Rs[i].T)
        pts.append(P[1:-1, 1:-1].reshape(-1, 3) @ Rs[i].T)  # camera-centred, world-oriented
    n = np.concatenate(ns)
    p = np.concatenate(pts)
    # Vertical axis: phones are held upright (portrait or landscape), so one camera axis stays
    # near-vertical; it keeps its world direction while the user turns, the other rotates with
    # yaw, so the vertical one has the longest frame-averaged vector.
    cands = []
    for ax in (0, 1):
        v = np.array([R[:, ax] for R in Rs])
        v = v * np.sign(v @ v[0])[:, None]
        m = v.mean(0)
        cands.append((np.linalg.norm(m), m / np.linalg.norm(m)))
    _, u = max(cands, key=lambda c: c[0])
    # refine with surface normals parallel to it (floor, ceiling, table tops)
    c = n @ u
    sel = np.abs(c) > np.cos(np.deg2rad(20))
    if sel.sum() > 50:
        u2 = np.median(n[sel] * np.sign(c[sel])[:, None], 0)
        u = u2 / np.linalg.norm(u2)
    # sign: more horizontal structure lies below the camera (floor + furniture tops) than above
    hs = np.abs(n @ u) > np.cos(np.deg2rad(15))
    h = p[hs] @ u
    below = np.sum((h < -0.3) & (h > -2.0))
    above = np.sum((h > 0.3) & (h < 2.0))
    up = u if below >= above else -u
    return up


def _align_up(up: np.ndarray) -> np.ndarray:
    """Rotation taking `up` to +y."""
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(up, y)
    s, c = np.linalg.norm(v), float(up @ y)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def fragment_capture(rec: dict, da: dict[str, np.ndarray], log=print):
    """One SfM fragment -> metric, gravity-aligned PseudoCapture (yaw/translation arbitrary)."""
    w, h = rec["wh"]
    f, cx, cy = rec["params"][:3]
    sx, sy = DEPTH_W / w, DEPTH_H / h
    K = np.array([[f * sx, 0, cx * sx], [0, f * sy, cy * sy], [0, 0, 1.0]])
    Rs, ts, ratios, names = [], [], [], []
    for im in sorted(rec["images"], key=lambda im: im["name"]):
        Rcw, tcw = im["R"], im["t"]
        uv, z = im["uv"], im["z"]
        if len(z) < 20:
            continue
        d = da[Path(im["name"]).stem]
        u = np.clip((uv[:, 0] * sx).astype(int), 0, DEPTH_W - 1)
        v = np.clip((uv[:, 1] * sy).astype(int), 0, DEPTH_H - 1)
        r = d[v, u] / z
        r = r[np.isfinite(r) & (r > 0)]
        if len(r) < 20:
            continue
        Rs.append(Rcw.T)
        ts.append(-Rcw.T @ tcw)
        names.append(Path(im["name"]).stem)
        ratios.append(float(np.median(r)))  # DA metres per SfM unit, this frame
    if len(names) < 5:
        return None, None
    ratios = np.array(ratios)
    s = float(np.median(ratios)) / DA_SCALE_BIAS  # metres per SfM unit
    spread = float(np.median(np.abs(np.log(ratios / np.median(ratios)))) * 1.4826)
    depths, confs = [], []
    for nm, k in zip(names, ratios):
        d = (da[nm] * (s / k)).astype(np.float32)  # DA rescaled to agree with metric SfM at sparse pts
        gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
        confs.append(np.where(np.hypot(gx, gy) > 0.15 * d, 0, 2).astype(np.uint8))  # DA blurs edges
        depths.append(d)
    Rs = np.array(Rs)
    ts = np.array(ts) * s
    G = _align_up(_gravity_from_normals(Rs, depths, K))
    pc = PseudoCapture(np.einsum("ij,njk->nik", G, Rs), ts @ G.T, K, depths, confs,
                       np.array([int(n) for n in names], float), names)
    pc = _refine_gravity(pc)
    stats = {"n": len(names), "scale_m_per_unit": s, "per_frame_log_spread": spread,
             "first": names[0], "last": names[-1], "focal_px": float(f)}
    return pc, stats


def _plan_raster(xy: np.ndarray, lo: np.ndarray, shape, res: float) -> np.ndarray:
    img = np.zeros(shape, np.float32)
    ij = np.floor((xy - lo) / res).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] < shape[0])
    np.add.at(img, (ij[ok, 1], ij[ok, 0]), 1.0)
    return cv2.GaussianBlur(np.minimum(img, 5), (0, 0), 1.5)


def _wall_xy(pc: PseudoCapture, idx=None):
    """Plan-view wall points + normals of a (sub)capture, for fragment registration."""
    from scan2plan.lidar.fusion import fuse
    from scan2plan.lidar.planes import estimate_levels

    kf = np.arange(pc.n_frames) if idx is None else idx
    c = fuse(pc, keyframes=kf, voxel=0.04)
    lv = estimate_levels(c.horiz, c.cam_y) if len(c.horiz) > 100 else None
    fy = lv.floor_y if lv else float(np.percentile(c.cam_y, 50) - 1.4)
    m = (c.vert[:, 1] > fy + 0.3) & (c.vert[:, 1] < fy + 2.0)
    return c.vert[m][:, [0, 2]], c.vert_n[m], fy


def stitch_fragments(frags: list[tuple[PseudoCapture, dict]], log=print) -> tuple[PseudoCapture, list[dict]]:
    """Place fragments (temporal order) into the first fragment's frame.

    Each fragment is metric and gravity-aligned, so the unknown is a yaw + plan translation
    (+ floor offset). Yaw: fragments' Manhattan axes must agree (4-fold ambiguity), the
    choice resolved by camera-heading continuity across the gap. Translation: FFT
    correlation of plan-view wall rasters within a window around the camera-continuity
    prior. Fragments whose best correlation is weak are dropped (reported)."""
    from scipy.signal import fftconvolve

    from scan2plan.lidar.planes import dominant_angle

    res = 0.04
    base, bst = frags[0]
    R_all, t_all = [base.rotations], [base.positions]
    D, C, N = list(base.depths), list(base.confs), list(base.names)
    wxy, wn, fy0 = _wall_xy(base)
    th0 = dominant_angle(wn)
    report = [{"fragment": 0, "n": bst["n"], "placed": True, "score": 1.0}]
    last_R, last_t = base.rotations[-1], base.positions[-1]
    for fi, (pc, st) in enumerate(frags[1:], start=1):
        bxy, bn, fyb = _wall_xy(pc)
        if len(bxy) < 200:
            report.append({"fragment": fi, "n": st["n"], "placed": False, "reason": "too few wall points"})
            continue
        thb = dominant_angle(bn)
        # heading of the last camera before the gap / first camera after it
        h_prev = np.arctan2(last_R[0, 2], last_R[2, 2])
        h_next = np.arctan2(pc.rotations[0][0, 2], pc.rotations[0][2, 2])
        best = None
        for k in range(4):
            yaw = (th0 - thb) + k * np.pi / 2  # plan-frame rotation applied to fragment
            # world yaw about +y: x' = c x + s z ... use 2D rotation on (x, z)
            c_, s_ = np.cos(yaw), np.sin(yaw)
            R2 = np.array([[c_, -s_], [s_, c_]])
            hd = np.angle(np.exp(1j * (h_next - yaw - h_prev)))  # heading change over the gap
            q = bxy @ R2.T
            cam0 = pc.positions[0][[0, 2]] @ R2.T
            prior = last_t[[0, 2]]
            lo = np.minimum(wxy.min(0), q.min(0) - cam0 + prior) - 2.0
            hi = np.maximum(wxy.max(0), q.max(0) - cam0 + prior) + 2.0
            shape = tuple((np.ceil((hi - lo) / res).astype(int) + 1)[::-1])
            A = _plan_raster(wxy, lo, shape, res)
            B = _plan_raster(q - cam0 + prior, lo, shape, res)
            corr = fftconvolve(A, B[::-1, ::-1], mode="same")
            cy_, cx_ = np.array(shape) // 2
            win = int(2.0 / res)  # +-2 m around the continuity prior
            sub = corr[cy_ - win:cy_ + win + 1, cx_ - win:cx_ + win + 1]
            j = np.unravel_index(np.argmax(sub), sub.shape)
            dy, dx = (j[0] - win) * res, (j[1] - win) * res
            norm = np.sqrt((A * A).sum() * (B * B).sum()) + 1e-9
            score = float(sub[j] / norm)
            # combine: correlation, penalised by implausible heading jumps over the gap
            gap_s = max(int(pc.names[0]) - int(N[-1]), 1) / 30.0
            pen = max(0.0, abs(hd) - np.deg2rad(60) * gap_s - np.deg2rad(30))
            total = score - 0.2 * pen
            if best is None or total > best[0]:
                best = (total, score, yaw, np.array([dx, dy]) + prior - cam0, k, hd)
        total, score, yaw, t2, k, hd = best
        placed = score > 0.12
        report.append({"fragment": fi, "n": st["n"], "placed": bool(placed), "score": round(score, 3),
                       "yaw_deg": round(float(np.degrees(yaw)), 1), "heading_jump_deg": round(float(np.degrees(hd)), 1)})
        if not placed:
            continue
        c_, s_ = np.cos(yaw), np.sin(yaw)
        Ry = np.array([[c_, 0, -s_], [0, 1, 0], [s_, 0, c_]])  # (x,z) rotation by yaw about +y
        Rn = np.einsum("ij,njk->nik", Ry, pc.rotations)
        tn = pc.positions @ Ry.T
        tn[:, 0] += t2[0]
        tn[:, 2] += t2[1]
        tn[:, 1] += fy0 - fyb
        R_all.append(Rn)
        t_all.append(tn)
        D += pc.depths
        C += pc.confs
        N += pc.names
        # grow the reference map with the placed fragment
        q = bxy @ np.array([[c_, -s_], [s_, c_]]).T + t2
        wxy = np.concatenate([wxy, q])
        last_R, last_t = Rn[-1], tn[-1]
    out = PseudoCapture(np.concatenate(R_all), np.concatenate(t_all), base.K_depth, D, C,
                        np.array([float(n) for n in N]), N)
    for r in report:
        log(f"[video] fragment {r}")
    return out, report


def _refine_gravity(pc: PseudoCapture, iters: int = 2) -> PseudoCapture:
    """Refine 'up' with the normal of the dominant near-horizontal surface (floor)."""
    from scan2plan.lidar.fusion import depth_normals

    for _ in range(iters):
        K = pc.K_depth
        vv, uu = np.mgrid[0:DEPTH_H:4, 0:DEPTH_W:4]
        ns = []
        for i in range(0, pc.n_frames, max(pc.n_frames // 60, 1)):
            d = pc.depths[i][::4, ::4]
            P = np.stack([(uu - K[0, 2]) / K[0, 0] * d, (vv - K[1, 2]) / K[1, 1] * d, d], 2)
            n = depth_normals(P)[1:-1, 1:-1].reshape(-1, 3) @ pc.rotations[i].T
            ns.append(n[np.abs(n[:, 1]) > np.cos(np.deg2rad(20))])
        n = np.concatenate(ns)
        n *= np.sign(n[:, 1:2])
        if len(n) < 50:
            break
        up = np.median(n, 0)
        up /= np.linalg.norm(up)
        if up[1] < np.cos(np.deg2rad(10)):
            break  # refinement only corrects small tilts
        G = _align_up(up)
        pc.rotations = np.einsum("ij,njk->nik", G, pc.rotations)
        pc.positions = pc.positions @ G.T
    return pc


def _apply(pc: PseudoCapture, R: np.ndarray, t: np.ndarray) -> PseudoCapture:
    return PseudoCapture(np.einsum("ij,njk->nik", R, pc.rotations), pc.positions @ R.T + t, pc.K_depth,
                         pc.depths, pc.confs, pc.timestamps, pc.names)


def _concat(parts: list[PseudoCapture]) -> PseudoCapture:
    return PseudoCapture(np.concatenate([p.rotations for p in parts]), np.concatenate([p.positions for p in parts]),
                         parts[0].K_depth, sum((p.depths for p in parts), []), sum((p.confs for p in parts), []),
                         np.concatenate([p.timestamps for p in parts]), sum((p.names for p in parts), []))


def chain_fragments(frags, work: Path, da: dict, log=print):
    """Chain SfM fragments with DA3 bridge transforms across each gap (fallback: wall-raster
    correlation stitch). Result is in fragment 0's gravity-aligned metric frame."""
    from scan2plan.video.bridge import bridge_transform, bridge_windows, run_da3_windows

    all_names = sorted(da.keys())
    wins = bridge_windows([pc.names for pc, _ in frags], all_names)
    orient = getattr(chain_fragments, "orient", 0)
    fdir = work / f"frames_o{orient}" if orient else work / "frames"
    bridges = run_da3_windows(wins, fdir, work / f"o{orient}", log=log)
    placed = [frags[0][0]]
    report = [{"fragment": 0, "n": frags[0][1]["n"], "placed": True, "via": "reference"}]
    prev = frags[0][0]
    for k, ((pc, st), br) in enumerate(zip(frags[1:], bridges), start=1):
        cands = []
        T, q = (None, {"ok": False, "reason": "DA3 failed"}) if br is None else bridge_transform(br, prev, pc, da, orient)
        # score against the recent map only (last 3 placed fragments): O(n) instead of O(n^2)
        map_pc = _concat(placed[-3:])
        if T is not None and q.get("gap_translation_m", 99) < 5:
            c = _apply(pc, *T)
            cands.append((_overlap_score(map_pc, c), "da3_bridge", c))
        stitched, rep = stitch_fragments([(prev, {"n": prev.n_frames}), (pc, st)], log=lambda m: None)
        if rep[-1].get("placed", False):
            c = PseudoCapture(stitched.rotations[prev.n_frames:], stitched.positions[prev.n_frames:], pc.K_depth,
                              pc.depths, pc.confs, pc.timestamps, pc.names)
            cands.append((_overlap_score(map_pc, c), "wall_correlation", c))
        if not cands:
            report.append({"fragment": k, "n": st["n"], "placed": False, "bridge": q})
            continue
        # DA3 bridge is trusted when its anchor frames agree with both fragments (measured:
        # 0.6-1.3 deg median relative-rotation error on upright frames); wall correlation is a
        # fallback because repeated room geometry makes its peak ambiguous.
        da3_ok = T is not None and q.get("anchor_rot_spread_deg", 99) < 5 and q.get("anchor_rot_spread_b_deg", 99) < 5
        alt = {c[1]: c[0] for c in cands}
        if da3_ok:
            score, via, nxt = next(c for c in cands if c[1] == "da3_bridge")
            score = max(score, 0.2)
        else:
            score, via, nxt = max(cands, key=lambda c: c[0])
        report.append({"fragment": k, "n": st["n"], "placed": score > 0.15, "via": via, "overlap": score, "bridge": q,
                       "alternatives": {v: round(sc, 3) for sc, v, _ in cands}})
        if score <= 0.15:
            continue
        placed.append(nxt)
        prev = nxt
    for r in report:
        log(f"[video] fragment {k_short(r)}")
    return _concat(placed), report


def _overlap_score(map_pc: PseudoCapture, frag: PseudoCapture, res: float = 0.05) -> float:
    """Fraction of the fragment's plan-view wall evidence that lands on the map's walls, with
    a penalty for fragment walls crossing the map's observed free space."""
    a, _, _ = _wall_xy(map_pc)
    b, _, _ = _wall_xy(frag)
    if len(a) < 50 or len(b) < 50:
        return 0.0
    lo = np.minimum(a.min(0), b.min(0)) - 1
    hi = np.maximum(a.max(0), b.max(0)) + 1
    shape = tuple((np.ceil((hi - lo) / res).astype(int) + 1)[::-1])
    A = _plan_raster(a, lo, shape, res) > 0.05
    A = cv2.dilate(A.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ij = np.floor((b - lo) / res).astype(int)
    return float(A[ij[:, 1], ij[:, 0]].mean())


def k_short(r: dict) -> dict:
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}


# Image rotations: code -> (cv2 rotate flag or None, camera-frame matrix M with p_rot = M p)
ROT = {
    0: (None, np.eye(3)),
    1: (cv2.ROTATE_90_CLOCKWISE, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])),
    2: (cv2.ROTATE_180, np.diag([-1, -1, 1.0])),
    3: (cv2.ROTATE_90_COUNTERCLOCKWISE, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]])),
}
UNROT = {0: None, 1: cv2.ROTATE_90_COUNTERCLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_CLOCKWISE}


def detect_orientation(frags) -> int:
    """Rotation code that makes frames upright, voted over fragments: world up expressed in
    camera coordinates should be -y (image up) after rotation."""
    votes = np.zeros(4)
    for pc, st in frags:
        up_cam = np.median(np.array([R.T @ np.array([0, 1.0, 0]) for R in pc.rotations]), 0)
        for code, (_, M) in ROT.items():
            votes[code] += st["n"] * float(-(M @ up_cam)[1])
    return int(np.argmax(votes))


def upright_frames(frames: list[Path], code: int, out: Path) -> list[Path]:
    if code == 0:
        return frames
    out.mkdir(parents=True, exist_ok=True)
    res = []
    for p in frames:
        q = out / p.name
        if not q.exists():
            cv2.imwrite(str(q), cv2.rotate(cv2.imread(str(p)), ROT[code][0]), [cv2.IMWRITE_JPEG_QUALITY, 95])
        res.append(q)
    return res


def video_cache_dir(video: Path) -> Path:
    h = hashlib.sha1(f"{video.resolve()}:{video.stat().st_size}".encode()).hexdigest()[:10]
    return Path("data/cache/video") / f"{video.stem}_{h}"


def run_video(src: Path, meta: dict, drift: bool = True, log=print):
    from scan2plan.lidar.drift import correct_drift
    from scan2plan.lidar.fusion import fuse, select_keyframes
    from scan2plan.lidar.pipeline import LidarResult
    from scan2plan.lidar.planes import dominant_angle, estimate_levels, rot2
    from scan2plan.lidar.rooms import segment_rooms, wall_band
    from scan2plan.lidar.to_plan import lidar_to_plan

    T = {}
    t0 = time.time()
    video = find_video(Path(src))
    work = video_cache_dir(video)
    frames = extract_frames(video, work / "frames")
    T["decode_s"] = time.time() - t0
    log(f"[video] {len(frames)} frames from {video.name} -> {work}")
    t1 = time.time()
    recs = run_sfm(work / "frames", work, log=log)
    T["sfm_s"] = time.time() - t1
    t1 = time.time()
    da = depth_cache(frames, work, log=log)
    T["depth_s"] = time.time() - t1
    def build(da_):
        out = []
        for rec in recs:
            pc_f, st = fragment_capture(rec, da_, log=log)
            if pc_f is not None:
                out.append((pc_f, st))
        return out

    frags = build(da)
    orient = detect_orientation(frags) if frags else 0
    if orient != 0:
        # monocular depth models expect upright images: re-run on rotated frames
        log(f"[video] frames are stored rotated (code {orient}); re-running depth on upright frames")
        t1 = time.time()
        da = depth_cache(frames, work, log=log, orient=orient)
        T["depth_s"] += time.time() - t1
        frags = build(da)
    if not frags:
        raise RuntimeError("video tier: no usable SfM fragment (too little texture / too fast motion)")
    chain_fragments.orient = orient
    pc, frag_report = chain_fragments(frags, work, da, log=log)
    n_frames_total = len(frames)
    vstats = {"fragments": [st for _, st in frags], "stitch": frag_report,
              "registered_frames": int(pc.n_frames), "decoded_frames": n_frames_total,
              "da_scale_bias": DA_SCALE_BIAS}
    (work / "video_stats.json").write_text(json.dumps(vstats, indent=1, default=float))
    np.savez(work / "stitched_poses.npz", R=pc.rotations, t=pc.positions, names=np.array(pc.names))
    rot, pos = pc.rotations, pc.positions
    dres = None
    allk = np.arange(pc.n_frames)
    if drift:
        coarse = fuse(pc, keyframes=allk, voxel=0.03)
        lv = estimate_levels(coarse.horiz, coarse.cam_y)
        t1 = time.time()
        dres = correct_drift(pc, lv.floor_y, wall_band(lv.floor_y, lv.ceiling_y))
        rot, pos = dres.rotations, dres.positions
        T["drift_s"] = time.time() - t1
    t1 = time.time()
    cloud = fuse(pc, rotations=rot, positions=pos, keyframes=allk, voxel=0.02, min_conf=2)
    level = estimate_levels(cloud.horiz, cloud.cam_y)
    theta = dominant_angle(cloud.vert_n)
    R2 = rot2(theta)
    v = cloud.vert
    seg = segment_rooms(v[:, [0, 2]] @ R2.T, cloud.vert_n @ R2.T, v[:, 1], cloud.cam_xz @ R2.T,
                        cloud.fan_range, R2, level.floor_y, level.ceiling_y)
    T["layout_s"] = time.time() - t1
    T["total_s"] = time.time() - t0
    res = LidarResult(pc, cloud, level, theta, R2, seg, dres, T)
    meta.update({"n_frames_used": pc.n_frames, "video": str(video), "sfm": vstats})
    plan = lidar_to_plan(res, meta)
    plan.tier = "video"
    _video_error_model(plan)
    plan.notes.append("Video tier: poses from COLMAP SfM on RGB only; metric scale from Depth-Anything-V2 "
                      f"metric-indoor (bias-corrected x{1 / DA_SCALE_BIAS:.3f}); intervals include a "
                      f"{VIDEO_SCALE_REL:.0%} relative scale term.")
    log(f"[video] {len(seg.polygons)} rooms, {len(seg.openings)} openings in {T['total_s']:.0f} s")
    return plan, {"lidar": res, "video": True, "frames_dir": str(work / "frames"), "orient": orient}


def _video_error_model(plan, face_floor: float = 0.02):
    """Widen LiDAR-style sigmas for monocular depth: per-face noise floor + global scale."""
    from scan2plan.measure import Measurement

    for r in plan.rooms:
        for e in r.edges:
            e.sigma_face = float(np.hypot(e.sigma_face, face_floor))
        # scale error is fully correlated across a room: add it to every edge as a length term
        r.scale_rel = VIDEO_SCALE_REL
        if r.ceiling is not None:
            c = r.ceiling
            r.ceiling = Measurement(c.value, float(np.hypot(c.sigma * 2, VIDEO_SCALE_REL * c.value)),
                                    c.method.replace("lidar", "video"), c.unit, c.observed)
    for o in plan.openings:
        w = float(np.linalg.norm(o.b - o.a))
        o.sigma_width = float(np.sqrt(o.sigma_width**2 + 2 * face_floor**2 + (VIDEO_SCALE_REL * w) ** 2))
        o.method = o.method.replace("lidar", "video")
