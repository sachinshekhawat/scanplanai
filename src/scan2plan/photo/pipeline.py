"""Photo tier: a folder of per-room folders, 2-8 stills each (no depth, no poses) -> one
stitched whole-property Plan.

Per room
  1. Intrinsics: EXIF 35 mm-equivalent focal if present, else the iPhone 1x prior (26 mm eq.).
  2. Metric depth per photo: DA3METRIC-LARGE (child process), scale bias-corrected as in video.
  3. Relative poses: SIFT matches between every photo pair, lifted to 3D with the metric
     depth, RANSAC rigid fit (3-point Kabsch). Maximum spanning tree on inlier counts, then
     a few rounds of pose averaging over all accepted pairs.
  4. Gravity: vertical camera axis + floor/ceiling normals (as video tier).
  5. Layout: the same fusion -> walls -> openings -> polygon code as LiDAR, room = the
     segment containing the photo standpoints.
Property
  6. Stitch (scan2plan.photo.stitch): rooms are Manhattan-aligned; each room is placed by
     matching one of its openings to an opening of an already-placed room (width agreement,
     back-to-back across a wall), choosing the rotation/door pair that maximises shared-wall
     agreement and forbids overlap. Folder order = walking order (protocol) breaks ties.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from scan2plan.io.stray import DEPTH_H, DEPTH_W

IMG_EXT = {".jpg", ".jpeg", ".png", ".heic", ".JPG", ".JPEG", ".PNG", ".HEIC"}
F35_DEFAULT = 26.0  # iPhone 15 main camera, 35 mm equivalent
# monocular metric-depth scale 1-sigma for a room from n photos: capture-level bias spread (0.039)
# plus per-frame scatter (0.118) averaged over the room's photos (n ~ 4-6) -> ~0.06
PHOTO_SCALE_REL = 0.065
MAX_SIDE = 1024


@dataclass
class Photo:
    path: Path
    img: np.ndarray  # BGR, resized
    f_px: float  # focal at resized resolution


def list_rooms(root: Path) -> list[Path]:
    rooms = sorted(p for p in root.iterdir() if p.is_dir() and any(q.suffix in IMG_EXT for q in p.iterdir()))
    if not rooms and any(q.suffix in IMG_EXT for q in root.iterdir()):
        rooms = [root]  # single room folder
    return rooms


def _exif_f35(path: Path) -> float | None:
    try:
        from PIL import Image

        ex = Image.open(path).getexif()
        sub = ex.get_ifd(0x8769) if hasattr(ex, "get_ifd") else {}
        v = sub.get(41989) or ex.get(41989)  # FocalLengthIn35mmFilm
        return float(v) if v else None
    except Exception:
        return None


def load_photos(folder: Path) -> list[Photo]:
    out = []
    for p in sorted(q for q in folder.iterdir() if q.suffix in IMG_EXT):
        if p.suffix.lower() == ".heic":
            try:
                import pillow_heif

                pillow_heif.register_heif_opener()
                from PIL import Image

                img = cv2.cvtColor(np.array(Image.open(p).convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception:
                continue
        else:
            img = cv2.imread(str(p))
        if img is None:
            continue
        if img.shape[0] > img.shape[1]:
            pass  # portrait is fine; intrinsics follow the image
        s = MAX_SIDE / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        f35 = _exif_f35(p) or F35_DEFAULT
        # 35 mm equivalent is defined on the 43.27 mm diagonal
        diag = np.hypot(*img.shape[:2])
        out.append(Photo(p, img, f35 / 43.27 * diag))
    return out


def _sift_pairs(photos: list[Photo]):
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.01)
    feats = []
    for ph in photos:
        g = cv2.cvtColor(ph.img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(g, None)
        feats.append((np.array([k.pt for k in kp], np.float32).reshape(-1, 2), des))
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = {}
    for i in range(len(photos)):
        for j in range(i + 1, len(photos)):
            if feats[i][1] is None or feats[j][1] is None or len(feats[i][0]) < 10 or len(feats[j][0]) < 10:
                continue
            m = bf.knnMatch(feats[i][1], feats[j][1], k=2)
            good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.8 * b.distance]
            if len(good) < 15:
                continue
            pi = feats[i][0][[g.queryIdx for g in good]]
            pj = feats[j][0][[g.trainIdx for g in good]]
            F, inl = cv2.findFundamentalMat(pi, pj, cv2.USAC_MAGSAC, 1.5, 0.999)
            if inl is None:
                continue
            inl = inl.ravel().astype(bool)
            if inl.sum() >= 15:
                pairs[(i, j)] = (pi[inl], pj[inl])
    return pairs


def _lift(pts: np.ndarray, depth: np.ndarray, ph: Photo) -> tuple[np.ndarray, np.ndarray]:
    h, w = ph.img.shape[:2]
    sx, sy = DEPTH_W / w, DEPTH_H / h
    u = np.clip((pts[:, 0] * sx).astype(int), 0, DEPTH_W - 1)
    v = np.clip((pts[:, 1] * sy).astype(int), 0, DEPTH_H - 1)
    z = depth[v, u]
    X = np.stack([(pts[:, 0] - w / 2) / ph.f_px * z, (pts[:, 1] - h / 2) / ph.f_px * z, z], 1)
    return X, (z > 0.2) & (z < 8)


def _kabsch(A, B):
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((B - mb).T @ (A - ma))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, mb - R @ ma


def rigid_ransac(A, B, iters=500, thr_rel=0.06, rng=None):
    """Rigid B ~ R A + t with depth-proportional inlier threshold (monocular depth noise)."""
    rng = rng or np.random.default_rng(0)
    best = (None, None, np.zeros(len(A), bool))
    if len(A) < 6:
        return best
    thr = thr_rel * np.maximum(np.linalg.norm(A, axis=1), 0.5) + 0.03
    for _ in range(iters):
        s = rng.choice(len(A), 3, replace=False)
        R, t = _kabsch(A[s], B[s])
        inl = np.linalg.norm(A @ R.T + t - B, axis=1) < thr
        if inl.sum() > best[2].sum():
            best = (R, t, inl)
    if best[2].sum() >= 6:
        R, t = _kabsch(A[best[2]], B[best[2]])
        inl = np.linalg.norm(A @ R.T + t - B, axis=1) < thr
        best = (R, t, inl)
    return best


def _rays(pts: np.ndarray, ph: Photo) -> np.ndarray:
    h, w = ph.img.shape[:2]
    r = np.stack([(pts[:, 0] - w / 2) / ph.f_px, (pts[:, 1] - h / 2) / ph.f_px, np.ones(len(pts))], 1)
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def rotation_ransac(ri: np.ndarray, rj: np.ndarray, thr_deg: float = 1.0, iters: int = 300, rng=None):
    """Pure-rotation model (photos taken turning in place): ri ~ R rj. 2-point minimal sets."""
    rng = rng or np.random.default_rng(0)
    best = (None, np.zeros(len(ri), bool))
    if len(ri) < 8:
        return best
    c = np.cos(np.deg2rad(thr_deg))
    for _ in range(iters):
        s = rng.choice(len(ri), 2, replace=False)
        R, _ = _kabsch(rj[s], ri[s])
        inl = np.sum((rj @ R.T) * ri, 1) > c
        if inl.sum() > best[1].sum():
            best = (R, inl)
    if best[1].sum() >= 8:
        R, _ = _kabsch(rj[best[1]], ri[best[1]])
        best = (R, np.sum((rj @ R.T) * ri, 1) > c)
    return best


def room_poses(photos: list[Photo], depths: list[np.ndarray], log=print):
    """Camera->room-frame poses for photos connected by rigid 3D-3D matches (frame of photo 0
    of the largest connected component)."""
    pairs = _sift_pairs(photos)
    edges = []
    for (i, j), (pi, pj) in pairs.items():
        Xi, oki = _lift(pi, depths[i], photos[i])
        Xj, okj = _lift(pj, depths[j], photos[j])
        ok = oki & okj
        R, t, inl = rigid_ransac(Xj[ok], Xi[ok])  # maps cam j -> cam i
        Rr, inl_r = rotation_ransac(_rays(pi, photos[i]), _rays(pj, photos[j]))
        n_rig = int(inl.sum()) if R is not None else 0
        n_rot = int(inl_r.sum())
        # prefer the pure rotation (protocol: turn in place) unless the rigid model explains
        # clearly more matches (photos from a second standpoint)
        if n_rot >= 12 and n_rot >= 0.8 * n_rig:
            edges.append((n_rot, i, j, Rr, np.zeros(3)))
        elif n_rig >= 12:
            edges.append((n_rig, i, j, R, t))
    n = len(photos)
    # maximum spanning tree (Prim) from the best-connected photo
    deg = np.zeros(n)
    for w, i, j, _, _ in edges:
        deg[i] += w
        deg[j] += w
    root = int(np.argmax(deg)) if edges else 0
    Rw = {root: np.eye(3)}
    tw = {root: np.zeros(3)}
    while True:
        cand = [(w, i, j, R, t) for w, i, j, R, t in edges if (i in Rw) != (j in Rw)]
        if not cand:
            break
        w, i, j, R, t = max(cand, key=lambda e: e[0])
        if i in Rw:  # j -> i known: T_w<-j = T_w<-i * T_i<-j
            Rw[j] = Rw[i] @ R
            tw[j] = Rw[i] @ t + tw[i]
        else:  # i unknown, j known: T_i<-j inverse
            Ri = R.T
            ti = -R.T @ t
            Rw[i] = Rw[j] @ Ri
            tw[i] = Rw[j] @ ti + tw[j]
    used = sorted(Rw)
    log(f"[photo]   {len(pairs)} matched pairs, {len(edges)} rigid edges, {len(used)}/{n} photos connected")
    return used, Rw, tw, edges


def _metric_depths(photos: list[Photo], work: Path, log=print) -> list[np.ndarray]:
    import subprocess
    import sys

    work.mkdir(parents=True, exist_ok=True)
    f = work / "da3m_depth.npz"
    names = [f"{k:03d}_{p.path.stem}" for k, p in enumerate(photos)]
    if not f.exists():
        d = work / "imgs"
        d.mkdir(exist_ok=True)
        paths = []
        for nm, p in zip(names, photos):
            q = d / f"{nm}.jpg"
            cv2.imwrite(str(q), p.img)
            paths.append(str(q.resolve()))
        (work / "list.txt").write_text("\n".join(paths))
        r = subprocess.run([sys.executable, "-m", "scan2plan.learned.da3_worker", "metric", str(work / "list.txt"), str(f)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-1500:])
    z = np.load(f)
    from scan2plan.video.pipeline import DA_SCALE_BIAS

    out = []
    for nm, p in zip(names, photos):
        d = z[nm].astype(np.float32)
        # worker assumed f = 0.83 x long side; rescale to this photo's focal, then bias-correct
        d *= p.f_px / (0.83 * max(p.img.shape[:2]))
        out.append(d / DA_SCALE_BIAS)
    return out


def photo_room_capture(folder: Path, work: Path, log=print):
    from scan2plan.video.pipeline import PseudoCapture, _align_up, _gravity_from_normals

    photos = load_photos(folder)
    if not photos:
        return None, {"room": folder.name, "error": "no images"}
    depths = _metric_depths(photos, work, log=log)
    used, Rw, tw, edges = room_poses(photos, depths, log=log)
    h, w = photos[0].img.shape[:2]
    # all photos share the device camera; use photo 0 focal for the depth grid intrinsics
    f = photos[used[0]].f_px
    K = np.array([[f * DEPTH_W / w, 0, DEPTH_W / 2], [0, f * DEPTH_H / h, DEPTH_H / 2], [0, 0, 1.0]])
    Rs = np.array([Rw[i] for i in used])
    ts = np.array([tw[i] for i in used])
    ds = [depths[i] for i in used]
    confs = []
    for d in ds:
        gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
        confs.append(np.where(np.hypot(gx, gy) > 0.15 * d, 0, 2).astype(np.uint8))
    G = _align_up(_gravity_from_normals(Rs, ds, K))
    pc = PseudoCapture(np.einsum("ij,njk->nik", G, Rs), ts @ G.T, K, ds, confs,
                       np.arange(len(used), dtype=float), [photos[i].path.name for i in used])
    stats = {"room": folder.name, "n_photos": len(photos), "n_connected": len(used), "n_rigid_edges": len(edges)}
    return pc, stats


def run_photo(src: Path, meta: dict, log=print):
    from scan2plan.photo.stitch import stitch_rooms

    T = {}
    t0 = time.time()
    rooms = list_rooms(Path(src))
    log(f"[photo] {len(rooms)} room folders under {src}")
    cache = Path("data/cache/photo") / Path(src).resolve().name
    room_results = []
    for k, folder in enumerate(rooms):
        log(f"[photo] room {k + 1}/{len(rooms)}: {folder.name}")
        pc, st = photo_room_capture(folder, cache / folder.name, log=log)
        room_results.append((folder.name, pc, st))
    T["per_room_s"] = time.time() - t0
    t1 = time.time()
    plan, report = stitch_rooms(room_results, meta, log=log)
    T["stitch_s"] = time.time() - t1
    T["total_s"] = time.time() - t0
    plan.timing.update(T)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "photo_report.json").write_text(json.dumps(report, indent=1, default=float))
    return plan, {"photo": True, "rooms": room_results}
