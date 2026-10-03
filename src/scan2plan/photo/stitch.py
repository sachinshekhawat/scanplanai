"""Whole-property stitch for per-room captures (photo tier; works for any per-room input).

Each room arrives in its own gravity-aligned metric frame with unknown yaw and position.
1. Per room: fusion -> segmentation; keep the segment holding the camera standpoints.
   Boundary openings = gaps in the structural wall image along each polygon edge
   (DOOR_MIN..DOOR_MAX wide, wall on both sides).
2. Rooms are rotated into their own Manhattan frame, so only k*90 deg rotations remain.
3. Greedy placement in folder (walking) order. For a new room, every (its door, placed
   door, k) triple with compatible widths and opposite door normals defines a placement
   with the two doors back-to-back across a wall of thickness WALL_T. Score:
     + shared wall length (parallel edges facing each other at 0.05-0.35 m)
     - 10 x overlap area with placed rooms (rooms may not overlap)
     - width mismatch / sigma, + bonus for attaching to the previous folder's room.
   The best placement is kept; rooms with no admissible placement are placed by the best
   shared-wall placement, and flagged (adjacency unverified) - never silently.
"""
from __future__ import annotations

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Polygon

from scan2plan.plan import Plan, PlanEdge, PlanOpening, PlanRoom
from scan2plan.lidar.rooms import DOOR_MAX, DOOR_MIN

WALL_T = 0.12
SIGMA_PLACE = 0.05  # 1-sigma per-room placement error added to stitched coordinates
SIGMA_UNOBSERVED_EDGE = 0.6  # free-space-bounded edge in a sparsely photographed room (x 0.5..1.5)


def _room_from_capture(name, pc, log=print):
    from scan2plan.lidar.fusion import fuse
    from scan2plan.lidar.pipeline import LidarResult
    from scan2plan.lidar.planes import dominant_angle, estimate_levels, rot2
    from scan2plan.lidar.rooms import segment_rooms
    from scan2plan.lidar.to_plan import lidar_to_plan

    kf = np.arange(pc.n_frames)
    cloud = fuse(pc, keyframes=kf, voxel=0.02, zmax=6.0)
    level = estimate_levels(cloud.horiz, cloud.cam_y)
    theta = dominant_angle(cloud.vert_n)
    R2 = rot2(theta)
    v = cloud.vert
    seg = segment_rooms(v[:, [0, 2]] @ R2.T, cloud.vert_n @ R2.T, v[:, 1], cloud.cam_xz @ R2.T,
                        cloud.fan_range, R2, level.floor_y, level.ceiling_y, min_room_area=0.8, min_cam_px=1,
                        min_views=1)  # 2-8 photos: a cell seen by one photo is evidence enough
    res = LidarResult(pc, cloud, level, theta, R2, seg, None, {})
    plan = lidar_to_plan(res, {"name": name}, scale=1.0)
    if not plan.rooms:
        plan.rooms = _fallback_room(seg, level)
        if not plan.rooms:
            return None
    cam = cloud.cam_xz @ R2.T
    # main room = the one containing most standpoints (else the largest)
    from shapely.geometry import Point

    def hits(r):
        return sum(r.polygon.buffer(0.3).contains(Point(*c)) for c in cam)

    room = max(plan.rooms, key=lambda r: (hits(r), r.polygon.area))
    room.name = ""
    doors = boundary_openings(room, seg)
    return {"name": name, "room": room, "doors": doors, "seg": seg, "level": level}


def _fallback_room(seg, level):
    """Too few registered photos to close a room: use the axis-aligned rectangle of the largest
    free-space component, every edge unsnapped (4 cm face sigma + placement + scale)."""
    from scipy import ndimage as ndi
    from shapely.geometry import box

    from scan2plan.lidar.to_plan import CEILING_PRIOR
    from scan2plan.measure import SIGMA_UNSNAPPED, unobserved

    lab, n = ndi.label(seg.free)
    if n == 0:
        return []
    k = int(np.argmax(ndi.sum(seg.free, lab, range(1, n + 1)))) + 1
    ys, xs = np.nonzero(lab == k)
    if len(xs) * seg.grid.res ** 2 < 0.8:
        return []
    p = seg.grid.to_m(np.array([[np.percentile(xs, 2), np.percentile(ys, 2)], [np.percentile(xs, 98) + 1, np.percentile(ys, 98) + 1]]))
    poly = box(p[0, 0], p[0, 1], p[1, 0], p[1, 1])
    c = list(poly.exterior.coords)[:-1]
    edges = [PlanEdge(np.array(c[i]), np.array(c[(i + 1) % 4]), SIGMA_UNSNAPPED * 2, False) for i in range(4)]
    return [PlanRoom(poly, edges, unobserved(*CEILING_PRIOR, "prior: ceiling not observed (photo fallback room)"))]


def boundary_openings(room: PlanRoom, seg) -> list[dict]:
    """Door-sized gaps in structural wall evidence along each polygon edge."""
    g = seg.grid
    wall = seg.wall_img > 0
    out = []
    for e in room.edges:
        a, b = np.asarray(e.a), np.asarray(e.b)
        L = float(np.linalg.norm(b - a))
        if L < DOOR_MIN + 0.2:
            continue
        d = (b - a) / L
        nrm = np.array([-d[1], d[0]])
        c = room.polygon.centroid
        if np.dot(np.array([c.x, c.y]) - (a + b) / 2, nrm) > 0:
            nrm = -nrm  # outward normal
        t = np.arange(0, L, g.res)
        hit = np.zeros(len(t), bool)
        for off in np.arange(-0.06, 0.16, g.res):
            q = a + np.outer(t, d) + nrm * off
            ix, iy = g.idx(q)
            hit |= wall[iy, ix]
        # gaps
        gaps, i = [], 0
        while i < len(t):
            if not hit[i]:
                j = i
                while j < len(t) and not hit[j]:
                    j += 1
                gaps.append((i, j))
                i = j
            else:
                i += 1
        for i0, i1 in gaps:
            w = (i1 - i0) * g.res
            bounded = i0 > 0 and i1 < len(t)  # wall on both sides
            if DOOR_MIN <= w <= DOOR_MAX and bounded:
                p0 = a + d * t[i0]
                p1 = a + d * min(t[i1 - 1] + g.res, L)
                out.append({"a": p0, "b": p1, "width": w, "normal": nrm, "sigma": 0.04})
    return out


def _rot(k):
    return np.linalg.matrix_power(np.array([[0.0, -1.0], [1.0, 0.0]]), k % 4)


def _tf_room(r: dict, k: int, t: np.ndarray) -> dict:
    R = _rot(k)
    poly = affinity.affine_transform(r["room"].polygon, [R[0, 0], R[0, 1], R[1, 0], R[1, 1], t[0], t[1]])
    doors = [{**d, "a": R @ d["a"] + t, "b": R @ d["b"] + t, "normal": R @ d["normal"]} for d in r["doors"]]
    edges = [PlanEdge(R @ np.asarray(e.a) + t, R @ np.asarray(e.b) + t, e.sigma_face, e.snapped) for e in r["room"].edges]
    return {**r, "poly": poly, "doors_w": doors, "edges_w": edges, "k": k, "t": t}


def _shared_wall(poly: Polygon, others: list[Polygon]) -> float:
    s = 0.0
    ring = poly.exterior
    for o in others:
        band = o.exterior.buffer(0.35)
        inter = ring.intersection(band)
        if not inter.is_empty:
            s += inter.length
    return s


def _score(cand: dict, placed: list[dict], prev_name: str | None, attach_to: str | None, wmis: float):
    polys = [p["poly"] for p in placed]
    ov = sum(cand["poly"].intersection(p.buffer(-0.03)).area for p in polys)
    sw = _shared_wall(cand["poly"], polys)
    s = sw - 10.0 * ov - 2.0 * wmis
    if attach_to is not None and attach_to == prev_name:
        s += 0.5
    return s, ov, sw


def stitch_rooms(room_results, meta: dict, log=print):
    rooms = []
    report = {"rooms": [], "placements": []}
    for name, pc, st in room_results:
        if pc is None:
            report["rooms"].append({**st, "used": False})
            continue
        r = _room_from_capture(name, pc, log=log)
        if r is None:
            report["rooms"].append({**st, "used": False, "error": "no room polygon"})
            continue
        r["coverage"] = st.get("n_connected", 1) / max(st.get("n_photos", 1), 1)
        report["rooms"].append({**st, "used": True, "area_m2": round(r["room"].polygon.area, 2),
                                "n_doors": len(r["doors"]),
                                "door_widths": [round(d["width"], 2) for d in r["doors"]]})
        log(f"[photo] {name}: {r['room'].polygon.area:.1f} m2, doors {[round(d['width'], 2) for d in r['doors']]}")
        rooms.append(r)
    if not rooms:
        raise RuntimeError("photo tier: no room could be reconstructed")
    placed = [_tf_room(rooms[0], 0, np.zeros(2))]
    links = []
    for ri_new, r in enumerate(rooms[1:], start=1):
        best = None
        for k in range(4):
            R = _rot(k)
            for dj, d_new in enumerate(r["doors"]):
                n_new = R @ d_new["normal"]
                c_new = R @ ((d_new["a"] + d_new["b"]) / 2)
                for pi, p in enumerate(placed):
                    for di, d_old in enumerate(p["doors_w"]):
                        if np.dot(n_new, d_old["normal"]) > -0.9:
                            continue  # doors must face each other
                        wmis = abs(d_new["width"] - d_old["width"]) / np.hypot(d_new["sigma"], d_old["sigma"])
                        if wmis > 4:
                            continue
                        c_old = (d_old["a"] + d_old["b"]) / 2
                        t = c_old + d_old["normal"] * WALL_T - c_new
                        cand = _tf_room(r, k, t)
                        sc, ov, sw = _score(cand, placed, rooms[ri_new - 1]["name"], p["name"], wmis)
                        if ov > 0.3:
                            continue
                        if best is None or sc > best[0]:
                            best = (sc, cand, {"room": r["name"], "to": p["name"], "door_new": dj, "door_old": di,
                                               "k": k, "overlap_m2": round(ov, 3), "shared_wall_m": round(sw, 2),
                                               "width_mismatch_sigma": round(float(wmis), 2), "via": "door"})
        if best is None:
            # fallback: shared-wall placement next to the previous room (adjacency unverified)
            prev = placed[-1]
            for k in range(4):
                for side in range(4):
                    base = _tf_room(r, k, np.zeros(2))
                    pb, cb = prev["poly"].bounds, base["poly"].bounds
                    offs = [np.array([pb[2] - cb[0] + WALL_T, pb[1] - cb[1]]),
                            np.array([pb[0] - cb[2] - WALL_T, pb[1] - cb[1]]),
                            np.array([pb[0] - cb[0], pb[3] - cb[1] + WALL_T]),
                            np.array([pb[0] - cb[0], pb[1] - cb[3] - WALL_T])]
                    cand = _tf_room(r, k, offs[side])
                    sc, ov, sw = _score(cand, placed, None, None, 0)
                    if ov < 0.05 and (best is None or sc > best[0]):
                        best = (sc, cand, {"room": r["name"], "to": prev["name"], "k": k, "via": "shared_wall_fallback",
                                           "overlap_m2": round(ov, 3), "shared_wall_m": round(sw, 2)})
        sc, cand, info = best
        placed.append(cand)
        report["placements"].append(info)
        log(f"[photo] placed {info}")
        if info["via"] == "door":
            j_old = next(i for i, p in enumerate(placed) if p["name"] == info["to"])
            links.append((len(placed) - 1, j_old, info["door_new"], info["door_old"]))
    # build Plan
    prooms, openings = [], []
    for p in placed:
        room = p["room"]
        # an edge not bounded by an observed wall face only says "the room extends at least to
        # here": with few registered photos the true wall can be metres further out
        cov = p.get("coverage", 1.0)
        unobs = SIGMA_UNOBSERVED_EDGE * (1.5 - min(cov, 1.0))
        edges = [PlanEdge(e.a, e.b, float(np.hypot(e.sigma_face if e.snapped else max(e.sigma_face, unobs), SIGMA_PLACE)),
                          e.snapped) for e in p["edges_w"]]
        pr = PlanRoom(p["poly"], edges, room.ceiling, scale_rel=PHOTO_SCALE_REL_LOCAL)
        prooms.append(pr)
    used = set()
    for ri, rj, dj, di in links:
        d = placed[ri]["doors_w"][dj]
        sd = float(np.hypot(d["sigma"], PHOTO_SCALE_REL_LOCAL * d["width"]))
        openings.append(PlanOpening(np.asarray(d["a"]), np.asarray(d["b"]), sd, "door", (rj, ri), "photo:wall-gap"))
        used.add((ri, dj))
        used.add((rj, di))
    for ri, p in enumerate(placed):
        for dj, d in enumerate(p["doors_w"]):
            if (ri, dj) in used:
                continue
            sd = float(np.hypot(d["sigma"], PHOTO_SCALE_REL_LOCAL * d["width"]))
            openings.append(PlanOpening(np.asarray(d["a"]), np.asarray(d["b"]), sd, "door", (ri,), "photo:wall-gap (exterior/unmatched)"))
    low = [r["name"] for r in rooms if r.get("coverage", 1) < 0.5]
    notes = ([f"Photo registration covered < 50% of the photos in {low}: room extents there are lower bounds "
              f"with widened intervals."] if low else []) + ["Photo tier: per-room metric depth (DA3METRIC) + 3D-3D rigid photo registration; rooms stitched by "
             "door matching in folder order; placement error and monocular scale widen every interval."]
    fb = [pl["room"] for pl in report["placements"] if pl["via"] != "door"]
    if fb:
        notes.append(f"Adjacency NOT verified by a matched doorway for: {fb} (placed against a shared wall).")
    plan = Plan("photo", prooms, openings, {**meta, "rooms_in": [r["name"] for r in rooms]},
                {"method": "per-room capture; inter-room drift not applicable (rooms placed by door matching)",
                 "enabled": False}, {}, notes)
    return plan, report


from scan2plan.photo.pipeline import PHOTO_SCALE_REL as PHOTO_SCALE_REL_LOCAL  # noqa: E402
