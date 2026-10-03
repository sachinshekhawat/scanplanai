"""LiDAR result -> tier-independent Plan (per-room ceilings, edge errors, openings)."""
from __future__ import annotations

import numpy as np
import shapely

from scan2plan.lidar.pipeline import LidarResult
from scan2plan.measure import (SCALE_REL, SIGMA_LAYOUT, SIGMA_JAMB, SIGMA_JAMB_RAW, SIGMA_LEVEL_SYS, SIGMA_UNSNAPPED,
                               Measurement, face_sigma, unobserved)
from scan2plan.plan import Plan, PlanEdge, PlanOpening, PlanRoom

CEILING_PRIOR = (2.60, 0.30)  # residential prior when the ceiling was never observed


def _level_in(y: np.ndarray, centre: float, half: float = 0.04):
    sel = y[np.abs(y - centre) < half]
    if len(sel) < 30:
        return None
    med = float(np.median(sel))
    sel = sel[np.abs(sel - med) < 0.025]
    mad = float(np.median(np.abs(sel - med)) * 1.4826)
    return float(np.median(sel)), mad, len(sel)


def room_heights(res: LidarResult, poly, cam_hi: float):
    h = res.cloud.horiz
    xy = h[:, [0, 2]] @ res.R2.T
    minx, miny, maxx, maxy = poly.bounds
    bb = (xy[:, 0] > minx) & (xy[:, 0] < maxx) & (xy[:, 1] > miny) & (xy[:, 1] < maxy)
    inside = np.zeros(len(h), bool)
    inner = poly.buffer(-0.05)
    if bb.any() and not inner.is_empty:
        inside[bb] = shapely.contains_xy(inner, xy[bb, 0], xy[bb, 1])
    y = h[inside, 1]
    fl = _level_in(y, res.level.floor_y, 0.05)
    floor_y, fs, fn = fl if fl else (res.level.floor_y, res.level.floor_sigma, 1000)
    up = y[y > cam_hi + 0.15]
    cover_needed = 0.15 * poly.area / 0.015**2
    if len(up) > 100:
        hist, edges = np.histogram(up, bins=np.arange(up.min(), up.max() + 0.01, 0.01))
        c = edges[np.argmax(hist)] + 0.005
        cl = _level_in(y, c)
        if cl and cl[2] > cover_needed and 1.9 < cl[0] - floor_y < 5.0:
            cy, cs, cn = cl
            s = float(np.sqrt(cs**2 / max(cn / 20, 1) + fs**2 / max(fn / 20, 1) + 2 * SIGMA_LEVEL_SYS**2
                              + (SCALE_REL * (cy - floor_y)) ** 2))
            frac = cn * 0.015**2 / poly.area
            return Measurement(cy - floor_y, s, f"lidar:ceiling-plane minus floor-plane (ceiling coverage {frac:.0%})")
    return unobserved(*CEILING_PRIOR, "prior: ceiling not observed in this capture (scan the ceiling to measure)")


def _apply_scale(rooms, openings, k: float) -> None:
    from shapely import affinity

    for r in rooms:
        r.polygon = affinity.scale(r.polygon, k, k, origin=(0, 0))
        for e in r.edges:
            e.a, e.b = e.a * k, e.b * k
        if r.ceiling is not None and r.ceiling.observed:
            r.ceiling = Measurement(r.ceiling.value * k, r.ceiling.sigma, r.ceiling.method, r.ceiling.unit, True)
    for o in openings:
        o.a, o.b = o.a * k, o.b * k


def lidar_to_plan(res: LidarResult, capture_meta: dict, scale: float | None = None) -> Plan:
    """scale: metric correction of the sensor (default LIDAR_DEPTH_SCALE for the LiDAR tier)."""
    from scan2plan.io.stray import LIDAR_DEPTH_SCALE

    scale = LIDAR_DEPTH_SCALE if scale is None else scale
    seg = res.seg
    cam_hi = float(np.percentile(res.cloud.cam_y, 98))
    rooms: list[PlanRoom] = []
    label_to_idx = {}
    for lbl, poly in sorted(seg.polygons.items()):
        edges = []
        for e in seg.edges[lbl]:
            s = face_sigma(e["sigma"], e["support"]) if e["snapped"] else SIGMA_UNSNAPPED
            s = float(np.sqrt(s**2 + e.get("amb", 0.0) ** 2 + SIGMA_LAYOUT**2))
            edges.append(PlanEdge(np.array(e["a"]), np.array(e["b"]), s, bool(e["snapped"])))
        if len(edges) < 3:
            continue
        label_to_idx[lbl] = len(rooms)
        rooms.append(PlanRoom(poly, edges, room_heights(res, poly, cam_hi)))
    openings = []
    for o in seg.openings:
        rr = tuple(label_to_idx[r] for r in o.rooms if r in label_to_idx)
        if len(rr) < 2:
            continue
        sj = [SIGMA_JAMB if j else SIGMA_JAMB_RAW for j in o.jamb_refined]
        s = float(np.sqrt(sj[0] ** 2 + sj[1] ** 2 + (SCALE_REL * o.width) ** 2))
        method = "lidar:jamb-to-jamb" if all(o.jamb_refined) else "lidar:freespace-chord"
        e = o.endpoints()
        openings.append(PlanOpening(e[0], e[1], s, o.kind, rr, method))
    drift = {"method": "submap pose graph: wall loop-closure + Manhattan yaw anchor + odometry prior + floor anchor",
             "enabled": res.drift is not None}
    if res.drift is not None:
        st = res.drift.stats
        drift.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()
                      if k != "match_median_residual_m"})
        h = st["match_median_residual_m"]
        drift["loop_residual_m_first_last"] = [round(h[0], 4), round(h[-1], 4)] if h else None
        drift["n_loop_matches_found"] = bool(h)
    if scale != 1.0:
        _apply_scale(rooms, openings, scale)
    notes = [f"LiDAR scale calibrated x{scale:.4f} (laser GT, benchmark/arkitscenes/calibration.md)."] if scale != 1.0 else []
    if res.level.ceiling_y is None:
        notes.append("Ceiling not observed: ceiling heights are priors (observed=false) with wide intervals.")
    return Plan("lidar", rooms, openings, capture_meta, drift, dict(res.timing), notes)
