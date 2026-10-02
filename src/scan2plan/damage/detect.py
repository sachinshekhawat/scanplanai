"""Per-surface damage regions with class and metric extent.

Hybrid detector (no labelled damage data exists for this task, so nothing is trained here):
 1. Views: keyframes every ~0.5 m / 20 deg with RGB + metric depth + pose (any tier).
 2. Surface map per view: each pixel's 3D point is assigned to a plan surface (wall face of a
    room polygon edge, floor, ceiling) if it lies within SURF_TOL of that plane.
 3. Candidates (classical, high recall):
      - crack : thin dark ridges (black-hat at 2 scales), elongated components
      - blotch: colour deviation (CIELAB) from the surface's median colour; stains, mould,
                peeling paint, holes
 4. Classification: CLIP ViT-B/32 zero-shot on each candidate crop against damage prompts and
    explicit negatives (shadows, sockets, frames, furniture, clean wall). Kept if a damage
    class wins with p >= P_MIN. The probability is reported as `confidence`.
 5. Metric extent: candidate pixels -> surface (u, v) coordinates (u along the wall, v height);
    detections of the same class on the same surface merge if their (u, v) boxes overlap.
    Extent sigma = across-view spread + depth error at that range.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SURF_TOL = 0.12
P_MIN = 0.40
CLASSES = {
    "crack": ["a crack in a wall", "a hairline crack in plaster"],
    "water_stain": ["a water stain on a wall", "a brown water damage stain on a ceiling"],
    "mold": ["black mold on a wall", "mould spots on a ceiling"],
    "peeling_paint": ["peeling paint", "flaking paint on a wall"],
    "hole": ["a hole in a wall", "a damaged hole in drywall"],
}
NEGATIVES = ["a clean painted wall", "a shadow on a wall", "a light switch or power socket", "a picture frame",
             "furniture", "a door", "a window", "a cable", "a reflection on a glossy surface", "tiles", "a grout line between tiles",
             "the edge of a cabinet", "a lamp", "a cushion", "a sofa", "a towel"]


@dataclass
class View:
    key: str
    bgr: np.ndarray  # image as stored (any size)
    depth: np.ndarray  # metres, DEPTH grid (192x256), aligned to bgr
    K: np.ndarray  # intrinsics of the depth grid
    R: np.ndarray  # cam -> world
    t: np.ndarray
    orient: int = 0  # rotation code to make the image upright (for CLIP)


@dataclass
class Surface:
    id: str
    kind: str  # wall | floor | ceiling
    room: str
    # wall: plan-frame line axis/c with extent [lo, hi] and inward normal sign
    axis: int = 0
    c: float = 0.0
    lo: float = 0.0
    hi: float = 0.0
    y: float = 0.0  # floor/ceiling height (world y)
    poly: object = None  # room polygon (plan) for floor/ceiling
    hits: list = field(default_factory=list)


def surfaces_from_doc(doc: dict, R2: np.ndarray, floor_y: float) -> list[Surface]:
    from shapely.geometry import Polygon

    out = []
    for r in doc["rooms"]:
        poly = Polygon(r["polygon"])
        H = (r["ceiling_height"] or {}).get("value", 2.6)
        for w in r["walls"]:
            a, b = np.array(w["start"]), np.array(w["end"])
            axis = 0 if abs(a[0] - b[0]) < abs(a[1] - b[1]) else 1
            c = (a[axis] + b[axis]) / 2
            lo, hi = sorted([a[1 - axis], b[1 - axis]])
            out.append(Surface(f"{w['id']}-S", "wall", r["id"], axis, c, lo, hi, floor_y, poly))
        out.append(Surface(f"{r['id']}-FLOOR", "floor", r["id"], y=floor_y, poly=poly))
        out.append(Surface(f"{r['id']}-CEIL", "ceiling", r["id"], y=floor_y + H, poly=poly))
    return out


def _surface_coords(P: np.ndarray, surfs: list[Surface], R2: np.ndarray):
    """Assign world points to surfaces. Returns (surface index or -1, u, v)."""
    import shapely

    xy = P[:, [0, 2]] @ R2.T
    sid = np.full(len(P), -1)
    best = np.full(len(P), SURF_TOL)
    u = np.zeros(len(P))
    v = np.zeros(len(P))
    for k, s in enumerate(surfs):
        if s.kind == "wall":
            d = np.abs(xy[:, s.axis] - s.c)
            along = xy[:, 1 - s.axis]
            m = (d < best) & (along > s.lo - 0.02) & (along < s.hi + 0.02) & (P[:, 1] > s.y - 0.05)
            sid[m], best[m], u[m], v[m] = k, d[m], along[m] - s.lo, P[m, 1] - s.y
        else:
            d = np.abs(P[:, 1] - s.y)
            m = d < best
            if m.any():
                inside = np.zeros(len(P), bool)
                inside[m] = shapely.contains_xy(s.poly, xy[m, 0], xy[m, 1])
                m &= inside
                sid[m], best[m], u[m], v[m] = k, d[m], xy[m, 0], xy[m, 1]
    return sid, u, v


def _candidates(bgr: np.ndarray, surf_mask: np.ndarray):
    """Classical candidates on surface pixels: list of (kind_hint, mask) at image resolution."""
    h, w = bgr.shape[:2]
    s = 960 / max(h, w)
    img = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr.copy()
    sm = cv2.resize(surf_mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    sm = cv2.erode(sm.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0  # stay off edges/corners
    out = []
    if sm.sum() < 2000:
        return out, img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if cv2.Laplacian(gray, cv2.CV_32F).var() < 3:  # heavily motion-blurred view
        return out, img
    # cracks: thin dark ridges
    bh = np.maximum(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))),
                    cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))))
    vals = bh[sm]
    mad = np.median(np.abs(vals - np.median(vals))) * 1.4826
    hi_t = np.median(vals) + 5 * mad + 3
    lo_t = np.median(vals) + 2.5 * mad + 1.5
    # hysteresis: weak ridge pixels connected (after 5 px linking) to strong ones
    weak = ((bh > lo_t) & sm).astype(np.uint8)
    weak = cv2.morphologyEx(weak, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    linked = cv2.dilate(weak, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n0, lab0 = cv2.connectedComponents(linked, connectivity=8)
    strong_ids = np.unique(lab0[(bh > hi_t) & sm])
    cr = (np.isin(lab0, strong_ids[strong_ids > 0]) & (weak > 0)).astype(np.uint8)
    cr_l = cv2.dilate(cr, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cr_l, connectivity=8)
    lab = np.where(cr > 0, lab, 0)
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        L = max(ww, hh)
        if L > 45 and a / max(ww * hh, 1) < 0.25 and a < 0.15 * L * L:  # long and thin
            ys, xs = np.nonzero(lab == i)
            P = np.stack([xs, ys], 1).astype(float)
            P -= P.mean(0)
            ev = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(P.T)), 0))
            # real cracks meander; grout lines, edges and cables are straight (minor/major ~ 0.01)
            if ev[0] / max(ev[1], 1e-6) > 0.06:
                out.append(("crack", lab == i))
    # blotches: colour deviation from the surface median (CIELAB), smoothed
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    med = np.median(lab_img[sm], axis=0)
    dev = np.linalg.norm((lab_img - med) * [0.6, 1.0, 1.0], axis=2)
    dev = cv2.GaussianBlur(dev, (0, 0), 3)
    dv = dev[sm]
    thr2 = np.median(dv) + 5 * np.median(np.abs(dv - np.median(dv))) * 1.4826 + 6
    bl = ((dev > thr2) & sm).astype(np.uint8)
    bl = cv2.morphologyEx(bl, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bl, connectivity=8)
    area_img = img.shape[0] * img.shape[1]
    for i in range(1, n):
        a = st[i, 4]
        if 0.002 * area_img < a < 0.25 * area_img:
            out.append(("blotch", lab == i))
    return out, img


def _crop(img, mask, pad=0.6):
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    ph, pw = int((y1 - y0) * pad) + 16, int((x1 - x0) * pad) + 16
    y0, y1 = max(0, y0 - ph), min(img.shape[0], y1 + ph)
    x0, x1 = max(0, x0 - pw), min(img.shape[1], x1 + pw)
    return img[y0:y1, x0:x1]


def classify_crops(crops: list[np.ndarray], work: Path) -> np.ndarray:
    """CLIP zero-shot in a child process (torch). Returns (N, n_classes+1) probs, last = negative."""
    if not crops:
        return np.zeros((0, len(CLASSES) + 1))
    work.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, c in enumerate(crops):
        p = work / f"crop_{k:04d}.jpg"
        cv2.imwrite(str(p), c)
        paths.append(str(p.resolve()))
    (work / "crops.txt").write_text("\n".join(paths))
    out = work / "clip_probs.npy"
    r = subprocess.run([sys.executable, "-m", "scan2plan.damage.clip_worker", str(work / "crops.txt"), str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])
    return np.load(out)


def detect(views: list[View], surfs: list[Surface], R2: np.ndarray, work: Path, log=print) -> list[dict]:
    from scan2plan.io.stray import DEPTH_H, DEPTH_W

    cands = []  # (view k, kind_hint, mask_lowres_surface_points...)
    crops = []
    for k, vw in enumerate(views):
        K = vw.K
        vv, uu = np.mgrid[0:DEPTH_H, 0:DEPTH_W]
        d = vw.depth
        ok = (d > 0.3) & (d < 4.0)
        P = np.stack([(uu - K[0, 2]) / K[0, 0] * d, (vv - K[1, 2]) / K[1, 1] * d, d], 2).reshape(-1, 3) @ vw.R.T + vw.t
        sid, su, sv = _surface_coords(P, surfs, R2)
        sid = np.where(ok.ravel(), sid, -1).reshape(DEPTH_H, DEPTH_W)
        found, img = [], None
        ids, cnt = np.unique(sid[sid >= 0], return_counts=True)
        for sk in ids[cnt > 0.04 * sid.size]:  # each visible surface separately (own colour stats)
            surf_mask = cv2.resize((sid == sk).astype(np.uint8), (vw.bgr.shape[1], vw.bgr.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            f_, img = _candidates(vw.bgr, surf_mask)
            found += f_
        for hint, m in found:
            # map candidate pixels to the depth grid -> surface coords
            md = cv2.resize(m.astype(np.uint8), (DEPTH_W, DEPTH_H), interpolation=cv2.INTER_NEAREST) > 0
            if hint == "crack":
                md = cv2.resize(cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)), (DEPTH_W, DEPTH_H),
                                interpolation=cv2.INTER_AREA) > 0
            idx = np.nonzero((md & (sid >= 0)).ravel())[0]
            if len(idx) < 3:
                continue
            ss = sid.ravel()[idx]
            s_major = int(np.bincount(ss).argmax())
            sel = idx[ss == s_major]
            crop = _crop(img, m)
            if vw.orient:
                from scan2plan.video.pipeline import ROT

                crop = cv2.rotate(crop, ROT[vw.orient][0])
            crops.append(crop)
            rng = float(np.median(d.ravel()[sel]))
            cands.append({"view": vw.key, "hint": hint, "surface": s_major, "u": su[sel], "v": sv[sel], "range": rng,
                          "px_frac": float(m.mean())})
    log(f"[damage] {len(views)} views, {len(cands)} candidates -> CLIP")
    probs = classify_crops(crops, work)
    names = list(CLASSES)
    dets = []
    acc_dir = work / "accepted"
    acc_dir.mkdir(parents=True, exist_ok=True)
    for f in acc_dir.glob("*.jpg"):
        f.unlink()
    for ci, (c, p) in enumerate(zip(cands, probs)):
        k = int(np.argmax(p))
        if k >= len(names) or p[k] < P_MIN:
            continue
        cls = names[k]
        if c["hint"] == "crack" and cls not in ("crack",):
            continue  # a thin-ridge candidate can only be a crack
        if c["hint"] == "blotch" and cls == "crack":
            continue
        dets.append({**c, "class": cls, "p": float(p[k])})
        cv2.imwrite(str(acc_dir / f"{cls}_{p[k]:.2f}_{c['view']}_{ci}.jpg"), crops[ci])
    # merge per (surface, class) by overlapping (u, v) boxes
    regions = []
    for dct in sorted(dets, key=lambda x: -x["p"]):
        box = np.array([dct["u"].min(), dct["v"].min(), dct["u"].max(), dct["v"].max()])
        hit = None
        for r in regions:
            if r["surface"] == dct["surface"] and r["class"] == dct["class"]:
                b = r["box"]
                if not (box[2] < b[0] - 0.05 or box[0] > b[2] + 0.05 or box[3] < b[1] - 0.05 or box[1] > b[3] + 0.05):
                    hit = r
                    break
        if hit is None:
            regions.append({"surface": dct["surface"], "class": dct["class"], "box": box, "boxes": [box],
                            "p": [dct["p"]], "views": [dct["view"]], "range": [dct["range"]],
                            "area_px": [len(dct["u"])], "uv": [np.stack([dct["u"], dct["v"]], 1)]})
        else:
            hit["box"] = np.r_[np.minimum(hit["box"][:2], box[:2]), np.maximum(hit["box"][2:], box[2:])]
            hit["boxes"].append(box)
            hit["p"].append(dct["p"])
            hit["views"].append(dct["view"])
            hit["range"].append(dct["range"])
            hit["uv"].append(np.stack([dct["u"], dct["v"]], 1))
    out = []
    for r in regions:
        bx = np.array(r["boxes"])
        wv = bx[:, 2] - bx[:, 0]
        hv = bx[:, 3] - bx[:, 1]
        W = float(np.median(wv))
        Hh = float(np.median(hv))
        # depth-grid pixel footprint at that range ~ range * 1/fx; LiDAR grid ~ 1.2 cm at 2 m
        pix = float(np.median(r["range"])) * 0.006 + 0.01
        sW = float(np.hypot(np.std(wv) if len(wv) > 1 else 0.3 * W, pix))
        sH = float(np.hypot(np.std(hv) if len(hv) > 1 else 0.3 * Hh, pix))
        uv = np.concatenate(r["uv"])
        cells = np.unique(np.floor(uv / 0.02).astype(int), axis=0)
        A = float(len(cells) * 0.02 ** 2) if r["class"] != "crack" else float(max(W, Hh) * 0.003)
        sA = float(np.hypot(A * 0.3, (W * sH + Hh * sW) * 0.5))
        s = surfs[r["surface"]]
        out.append({"surface": s.id, "surface_kind": s.kind, "room": s.room, "class": r["class"],
                    "confidence": float(np.mean(r["p"])), "n_views": len(set(r["views"])),
                    "width": (W, sW), "height": (Hh, sH), "area": (A, sA), "length": (float(np.hypot(W, Hh)), float(np.hypot(sW, sH))),
                    "uv_box": [float(x) for x in r["box"]], "surface_obj": s})
    log(f"[damage] {len(dets)} accepted detections -> {len(out)} regions")
    return out
