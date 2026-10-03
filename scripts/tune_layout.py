"""Iterative layout tuning with a train / validation split (scripts/tune_layout.py).

    python scripts/tune_layout.py train   # score current code on TRAIN walks
    python scripts/tune_layout.py val     # score on VALIDATION (run once per final candidate)

TRAIN: ARKitScenes dev visit 422382 + visits listed in data/arkitscenes/train_visits.txt.
VAL  : ARKitScenes visit 422386 (never tuned on) + the sample-data cross-session pair.
Per walk, the expensive part (fusion, drift, laser ICP, laser -> plan frame) is cached in
data/cache/tune/<walk>.pkl; an iteration re-runs only segmentation + scoring.
Score (lower is better): median |wall length error| over all GT-matched walls, reported with
coverage of the 95% intervals and the number of matched walls (so a change cannot win by
producing fewer, easier walls).
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
BASE = ROOT / "data" / "arkitscenes"
CACHE = ROOT / "data" / "cache" / "tune"


def walks(split):
    if split == "val":
        return [("422386", "42446527"), ("422386", "42446532")]
    out = [("422382", v) for v in ("42446533", "42446535", "42446536")]
    f = BASE / "train_visits.txt"
    if f.exists():
        for line in f.read_text().split("\n"):
            p = line.split()
            if len(p) >= 4 and (BASE / "laser" / p[0] / f"{p[1]}.ply").exists():
                out += [(p[0], v) for v in p[2:] if (BASE / "raw" / v / "lowres_wide.traj").exists()]
    return out


def prepared(visit, v):
    from arkitscenes_gt import icp_ipad_to_laser, laser_world

    from scan2plan.io.arkitscenes import load_arkitscenes
    from scan2plan.lidar.pipeline import run_lidar

    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{v}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    res = run_lidar(BASE / "raw" / v, log=lambda m: None)
    cap = load_arkitscenes(BASE / "raw" / v)
    L = laser_world(visit)
    T, q = icp_ipad_to_laser(cap, L)
    P = ((L.astype(np.float64) - T[:3, 3]) @ T[:3, :3]) @ cap.world_to_ours.T
    keep = np.random.default_rng(0).random(len(P)) < min(1.0, 3e6 / len(P))
    d = {"res": res, "P": P[keep].astype(np.float32), "icp": q}
    res.cap = None  # do not pickle file handles
    pickle.dump(d, open(f, "wb"))
    return d


def score_walk(d):
    import shapely

    from arkitscenes_gt import gt_face

    import scan2plan.lidar.rooms as RM
    from scan2plan.lidar.to_plan import lidar_to_plan
    from scan2plan.plan import to_document

    res = d["res"]
    c = res.cloud
    v = c.vert
    res.seg = RM.segment_rooms(v[:, [0, 2]] @ res.R2.T, c.vert_n @ res.R2.T, v[:, 1], c.cam_xz @ res.R2.T, c.fan_range,
                               res.R2, res.level.floor_y, res.level.ceiling_y, fans_high=getattr(c, "fan_high", None))
    doc = to_document(lidar_to_plan(res, {"name": "t", "tier": "lidar"}))
    P = d["P"].astype(np.float64)
    xy_all = P[:, [0, 2]] @ res.R2.T
    walls, ceil = [], []
    for room in doc["rooms"]:
        poly = shapely.Polygon(room["polygon"])
        inside = shapely.contains_xy(poly.buffer(-0.1), xy_all[:, 0], xy_all[:, 1])
        yr = P[inside, 1]
        fl = float(np.median(P[:, 1]))
        if len(yr) > 500:
            hb = np.arange(yr.min(), yr.max() + 0.005, 0.005)
            hh, _ = np.histogram(yr, hb)
            mid = (yr.min() + yr.max()) / 2
            fl = hb[np.argmax(hh * (hb[:-1] < mid))] + 0.0025
            ce = hb[np.argmax(hh * (hb[:-1] > mid))] + 0.0025
            ch = room["ceiling_height"]
            if ch["observed"] and abs(ch["value"] - (ce - fl)) < 0.3:
                ceil.append((ch["value"] - (ce - fl), ch["sigma"]))
        band = (P[:, 1] > fl + 0.3) & (P[:, 1] < fl + 1.8)
        xy = P[band][:, [0, 2]] @ res.R2.T
        W = room["walls"]
        faces = []
        for w in W:
            a, b = np.array(w["start"]), np.array(w["end"])
            ax = 0 if abs(a[0] - b[0]) < abs(a[1] - b[1]) else 1
            lo, hi = sorted([a[1 - ax], b[1 - ax]])
            faces.append(gt_face(xy, ax, (a[ax] + b[ax]) / 2, lo, hi) if hi - lo > 0.15 else None)
        for k, w in enumerate(W):
            fp, fn = faces[k - 1], faces[(k + 1) % len(W)]
            if fp is None or fn is None or w["length"]["value"] < 0.3:
                continue
            gt = abs(fn - fp)
            walls.append((w["length"]["value"] - gt, w["length"]["sigma"], gt))
    n_walls_total = sum(len(r["walls"]) for r in doc["rooms"])
    return {"walls": walls, "ceil": ceil, "n_rooms": len(doc["rooms"]), "n_walls_total": n_walls_total,
            "area": doc["property"]["floor_area_total"]["value"]}


def sample_pair():
    """Cross-session agreement on the sample flat (validation only)."""
    from scan2plan.benchmark.run import _tf_doc, align_docs, match_rooms, match_walls
    import scan2plan.lidar.rooms as RM
    from scan2plan.lidar.to_plan import lidar_to_plan
    from scan2plan.plan import to_document

    docs = {}
    for n in ("single_scan_with_ceiling", "single_scan_floor_only"):
        f = CACHE / f"sample_{n}.pkl"
        if f.exists():
            res = pickle.load(open(f, "rb"))
        else:
            from scan2plan.lidar.pipeline import run_lidar

            res = run_lidar(ROOT / "data" / "raw" / n, log=lambda m: None)
            res.cap = None
            pickle.dump(res, open(f, "wb"))
        c = res.cloud
        v = c.vert
        res.seg = RM.segment_rooms(v[:, [0, 2]] @ res.R2.T, c.vert_n @ res.R2.T, v[:, 1], c.cam_xz @ res.R2.T, c.fan_range,
                                   res.R2, res.level.floor_y, res.level.ceiling_y, fans_high=getattr(c, "fan_high", None))
        docs[n] = to_document(lidar_to_plan(res, {"name": n, "tier": "lidar"}))
    a, b = docs["single_scan_with_ceiling"], docs["single_scan_floor_only"]
    iou, k, t = align_docs(a, b)
    bb = _tf_doc(b, k, t)
    rows = []
    for i, j, _ in match_rooms(a, bb):
        for wa, wb in match_walls(a["rooms"][i], bb["rooms"][j]):
            L = wa["length"]["value"]
            dd = wb["length"]["value"] - L
            rows.append((abs(dd) <= max(0.01, 0.005 * L), dd, np.hypot(wa["length"]["sigma"], wb["length"]["sigma"])))
    return {"iou": iou, "n_walls": len(rows), "pass": float(np.mean([r[0] for r in rows])),
            "median_abs_cm": 100 * float(np.median([abs(r[1]) for r in rows])),
            "cov95": float(np.mean([abs(r[1]) <= 1.96 * r[2] for r in rows]))}


def summarize(results):
    W = [w for r in results.values() for w in r["walls"]]
    C = [c for r in results.values() for c in r["ceil"]]
    e = np.array([w[0] for w in W]) if W else np.zeros(0)
    s = np.array([w[1] for w in W]) if W else np.zeros(0)
    g = np.array([w[2] for w in W]) if W else np.zeros(0)
    ce = np.array([c[0] for c in C]) if C else np.zeros(0)
    cs = np.array([c[1] for c in C]) if C else np.zeros(0)
    return {"n_walls": int(len(e)), "median_abs_cm": round(100 * float(np.median(np.abs(e))), 2) if len(e) else None,
            "within_3pct": round(float(np.mean(np.abs(e) <= 0.03 * g)), 3) if len(e) else None,
            "within_5cm": round(float(np.mean(np.abs(e) <= 0.05)), 3) if len(e) else None,
            "cov95": round(float(np.mean(np.abs(e) <= 1.96 * s)), 3) if len(e) else None,
            "ceiling_n": int(len(ce)), "ceiling_median_abs_cm": round(100 * float(np.median(np.abs(ce))), 2) if len(ce) else None,
            "ceiling_cov95": round(float(np.mean(np.abs(ce) <= 1.96 * cs)), 3) if len(ce) else None}


def main(split="train", tag=""):
    res = {f"{vis}/{v}": score_walk(prepared(vis, v)) for vis, v in walks(split)}
    out = {"split": split, "tag": tag, "walks": {k: {kk: vv for kk, vv in r.items() if kk not in ("walls", "ceil")}
                                                 | {"n_gt_walls": len(r["walls"])} for k, r in res.items()},
           "summary": summarize(res)}
    if split == "val":
        out["sample_pair"] = sample_pair()
    log = ROOT / "benchmark" / "arkitscenes" / "tuning_log.jsonl"
    with open(log, "a") as f:
        f.write(json.dumps(out, default=float) + "\n")
    print(json.dumps({"split": split, "tag": tag, **out["summary"], **({"sample_pair": out["sample_pair"]} if split == "val" else {})},
                     default=float))


if __name__ == "__main__":
    main(*sys.argv[1:])
