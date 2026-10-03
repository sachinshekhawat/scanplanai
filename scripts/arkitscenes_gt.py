"""Score scan2plan against ARKitScenes laser-scanner ground truth (external benchmark).

ARKitScenes (Apple, CC BY-NC-SA 4.0) ships iPad Pro LiDAR walks (same sensor class and format
family as Stray Scanner) plus Faro laser scans of the same space (mm-level). Pipeline:
  1. laser scan -> ARKitScenes world: p_world = (p - T[3,:3]) @ T[:3,:3].T  (verified: this is the
     only convention of four that brings the scan within 0.3 m of the iPad cloud);
  2. per walk, point-to-plane ICP of the iPad depth cloud onto the laser cloud (5 coarse-to-fine
     gates, 0.5 -> 0.02 m) -> laser expressed in the pipeline's gravity-aligned plan frame;
  3. GT per predicted wall: the laser face = median offset of laser points within +-0.25 m of the
     predicted wall line, 0.3 .. 1.8 m above the laser floor, inside the wall's extent shrunk by 15%;
     GT wall length = distance between the GT faces of its two perpendicular neighbours;
  4. GT ceiling height = laser ceiling slab - laser floor slab inside the room polygon.
Every predicted number is compared with GT; interval coverage = |err| <= 1.96 sigma.

    python scripts/arkitscenes_gt.py [visit=422382] [videos...]
      -> benchmark/arkitscenes/gt_<visit>.json and report_<visit>.md
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BASE = ROOT / "data" / "arkitscenes"


def read_ply_xyz(path: Path, stride: int = 40) -> np.ndarray:
    """Binary little-endian PLY vertex reader (xyz + any per-vertex props), subsampled."""
    types = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4",
             "uint": "<u4", "short": "<i2", "ushort": "<u2"}
    with open(path, "rb") as f:
        props, n, in_vertex = [], 0, False
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if line.startswith("element"):
                in_vertex = line.startswith("element vertex")
                if in_vertex:
                    n = int(line.split()[-1])
            elif line.startswith("property") and in_vertex:
                props.append(line.split()[1])
            if line == "end_header":
                break
        dt = np.dtype([(f"p{i}", types[t]) for i, t in enumerate(props)])
        data = np.memmap(f, dtype=dt, mode="r", offset=f.tell(), shape=(n,))[::stride]
        return np.stack([data["p0"], data["p1"], data["p2"]], 1).astype(np.float64)


def laser_world(visit: str) -> np.ndarray:
    cache = BASE / "laser" / visit / "laser_world_s40.npy"
    if cache.exists():
        return np.load(cache)
    pts = []
    for ply in sorted((BASE / "laser" / visit).glob("*.ply")):
        T = np.loadtxt(ply.with_name(ply.stem + "_pose.txt"), delimiter=",")
        P = read_ply_xyz(ply)
        pts.append((P - T[3, :3]) @ T[:3, :3].T)
    L = np.concatenate(pts).astype(np.float32)
    np.save(cache, L)
    return L


def icp_ipad_to_laser(cap, L):
    import open3d as o3d

    G = cap.world_to_ours
    ip = np.concatenate([cap.backproject(i, stride=6) for i in range(0, cap.n_frames, 6)]) @ G  # ARKitScenes world
    a = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(L.astype(np.float64))).voxel_down_sample(0.02)
    b = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ip)).voxel_down_sample(0.02)
    a.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    T = np.eye(4)
    for th in (0.5, 0.25, 0.1, 0.05, 0.02):
        r = o3d.pipelines.registration.registration_icp(
            b, a, th, T, o3d.pipelines.registration.TransformationEstimationPointToPlane())
        T = r.transformation
    d = np.asarray(b.transform(T).compute_point_cloud_distance(a))
    return T, {"icp_rmse_m": float(r.inlier_rmse), "ipad_to_laser_median_m": float(np.median(d)),
               "frac_within_2cm": float((d < 0.02).mean())}


def gt_face(xy, axis, c, lo, hi):
    """Laser face position for a predicted wall line (axis=0: x=c, along y in [lo, hi])."""
    m = 0.15 * (hi - lo)
    sel = (np.abs(xy[:, axis] - c) < 0.25) & (xy[:, 1 - axis] > lo + m) & (xy[:, 1 - axis] < hi - m)
    v = xy[sel, axis]
    if len(v) < 50:
        return None
    h, e = np.histogram(v, bins=np.arange(c - 0.25, c + 0.255, 0.005))
    pk = e[np.argmax(np.convolve(h, np.ones(3), "same"))] + 0.0025
    w = v[np.abs(v - pk) < 0.015]
    return float(np.median(w)) if len(w) > 20 else None


def main(visit="422382", *videos):
    import shapely

    from scan2plan.cli import run
    from scan2plan.io.arkitscenes import load_arkitscenes
    from scan2plan.lidar.pipeline import run_lidar

    videos = videos or tuple(sorted(p.name for p in (BASE / "raw").iterdir() if p.is_dir()))
    L = laser_world(visit)
    out = {"dataset": "ARKitScenes raw (Validation), visit " + visit, "license": "CC BY-NC-SA 4.0", "walks": {}}
    for v in videos:
        rd = ROOT / "runs" / "arkit" / v
        if not (rd / "plan.json").exists():
            run(str(BASE / "raw" / v), "lidar", str(rd))
        doc = json.loads((rd / "plan.json").read_text())
        cap = load_arkitscenes(BASE / "raw" / v)
        T, q = icp_ipad_to_laser(cap, L)
        # laser -> iPad ARKitScenes world -> our gravity frame -> plan frame
        Li = (L.astype(np.float64) - T[:3, 3]) @ T[:3, :3]
        P = Li @ cap.world_to_ours.T
        R2 = run_lidar(BASE / "raw" / v, log=lambda m: None).R2
        walls, heights = [], []
        for room in doc["rooms"]:
            poly = shapely.Polygon(room["polygon"])
            xy_all = P[:, [0, 2]] @ R2.T
            inside = shapely.contains_xy(poly.buffer(-0.1), xy_all[:, 0], xy_all[:, 1])
            yr = P[inside, 1]
            if len(yr) > 500:
                hb = np.arange(yr.min(), yr.max() + 0.005, 0.005)
                hh, _ = np.histogram(yr, hb)
                mid = (yr.min() + yr.max()) / 2
                fl = hb[np.argmax(hh * (hb[:-1] < mid))] + 0.0025
                ce = hb[np.argmax(hh * (hb[:-1] > mid))] + 0.0025
                ch = room["ceiling_height"]
                heights.append({"room": room["id"], "gt_m": float(ce - fl), "pred_m": ch["value"], "sigma": ch["sigma"],
                                "observed": ch["observed"], "err_m": ch["value"] - float(ce - fl)})
            else:
                fl = float(np.median(P[:, 1]))
            band = (P[:, 1] > fl + 0.3) & (P[:, 1] < fl + 1.8)
            xy = P[band][:, [0, 2]] @ R2.T
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
                L_ = w["length"]
                walls.append({"room": room["id"], "wall": w["id"], "gt_m": gt, "pred_m": L_["value"], "sigma": L_["sigma"],
                              "err_m": L_["value"] - gt, "bounded_by_wall_face": w.get("bounded_by_wall_face")})
        e = np.array([x["err_m"] for x in walls]) if walls else np.zeros(0)
        s = np.array([x["sigma"] for x in walls]) if walls else np.zeros(0)
        summ = {"n_walls": int(len(e)),
                "median_abs_err_m": float(np.median(np.abs(e))) if len(e) else None,
                "within_1cm_or_0.5pct": float(np.mean([abs(x["err_m"]) <= max(0.01, 0.005 * x["gt_m"]) for x in walls])) if walls else None,
                "within_3pct": float(np.mean([abs(x["err_m"]) <= 0.03 * x["gt_m"] for x in walls])) if walls else None,
                "ci95_coverage": float(np.mean(np.abs(e) <= 1.96 * s)) if len(e) else None,
                "face_bounded_median_abs_err_m": float(np.median([abs(x["err_m"]) for x in walls if x["bounded_by_wall_face"]]))
                if any(x["bounded_by_wall_face"] for x in walls) else None}
        out["walks"][v] = {"registration": q, "walls": walls, "ceiling": heights, "summary": summ,
                           "n_rooms": len(doc["rooms"]), "floor_area_m2": doc["property"]["floor_area_total"]["value"]}
        print(v, q, summ, [(h["room"], round(h["gt_m"], 3), round(h["pred_m"], 3)) for h in heights], flush=True)
    # repeatability across the walks of the same room (largest room of each walk)
    areas = [w["floor_area_m2"] for w in out["walks"].values()]
    hts = [h["pred_m"] for w in out["walks"].values() for h in w["ceiling"] if h["observed"]]
    out["repeatability"] = {"floor_area_m2_per_walk": areas,
                            "ceiling_pred_spread_m": float(np.ptp(hts)) if hts else None}
    d = ROOT / "benchmark" / "arkitscenes"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"gt_{visit}.json").write_text(json.dumps(out, indent=1, default=float))
    lines = [f"# External ground truth: {out['dataset']}", "", "Laser-scanner GT vs scan2plan LiDAR tier "
             "(same pipeline, unchanged parameters).", "",
             "| walk | ICP rmse | walls | median abs err | <=1cm/0.5% | <=3% | face-bounded median | CI95 coverage |",
             "|---|---|---|---|---|---|---|---|"]
    for v, w in out["walks"].items():
        s_ = w["summary"]
        f = lambda x, p=1: "-" if x is None else (f"{x * 100:.{p}f}")  # noqa: E731
        lines.append(f"| {v} | {w['registration']['icp_rmse_m'] * 100:.1f} cm | {s_['n_walls']} | {f(s_['median_abs_err_m'])} cm | "
                     f"{f(s_['within_1cm_or_0.5pct'], 0)}% | {f(s_['within_3pct'], 0)}% | {f(s_['face_bounded_median_abs_err_m'])} cm | "
                     f"{f(s_['ci95_coverage'], 0)}% |")
    lines += ["", "## Ceiling height (GT = laser ceiling - laser floor inside room)", "", "| walk | room | GT (m) | pred (m) | err (cm) | sigma (cm) |", "|---|---|---|---|---|---|"]
    for v, w in out["walks"].items():
        for h in w["ceiling"]:
            lines.append(f"| {v} | {h['room']} | {h['gt_m']:.3f} | {h['pred_m']:.3f} | {h['err_m'] * 100:+.1f} | {h['sigma'] * 100:.1f} |")
    lines += ["", "## Per wall", "", "| walk | wall | GT (m) | pred (m) | err (cm) | sigma (cm) | face-bounded |", "|---|---|---|---|---|---|---|"]
    for v, w in out["walks"].items():
        for x in w["walls"]:
            lines.append(f"| {v} | {x['wall']} | {x['gt_m']:.3f} | {x['pred_m']:.3f} | {x['err_m'] * 100:+.1f} | {x['sigma'] * 100:.1f} | {x['bounded_by_wall_face']} |")
    (d / f"report_{visit}.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
