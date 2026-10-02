"""Tier-independent plan model -> output document (schema v1).

Every tier produces a `Plan`: rooms as rectilinear polygons with per-edge error, openings,
levels. This module turns a Plan into the published JSON (see schema/plan.schema.json),
including adjacency, surfaces and their areas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from scan2plan import __version__
from scan2plan.measure import TIER_K, Measurement, length_sigma, unobserved

SCHEMA_VERSION = "1.0.0"


@dataclass
class PlanEdge:
    a: np.ndarray  # (2,) plan coords
    b: np.ndarray
    sigma_face: float  # 1-sigma of this edge's face position (m)
    snapped: bool


@dataclass
class PlanOpening:
    a: np.ndarray
    b: np.ndarray
    sigma_width: float
    kind: str  # door | open_passage | window
    rooms: tuple  # room indices (0-based) it connects; one entry for exterior/window
    method: str


@dataclass
class PlanRoom:
    polygon: Polygon
    edges: list[PlanEdge]
    ceiling: Measurement | None = None  # height above floor
    name: str = ""
    scale_rel: float | None = None  # tier's relative scale 1-sigma (None = LiDAR default)


@dataclass
class Plan:
    tier: str
    rooms: list[PlanRoom]
    openings: list[PlanOpening]
    capture: dict
    drift: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)  # damage, flags, scope added downstream


def _classify(poly: Polygon, n_openings: int) -> str:
    minx, miny, maxx, maxy = poly.bounds
    w, h = sorted([maxx - minx, maxy - miny])
    if poly.area < 2.5:
        return "closet_or_wc"
    if h / max(w, 1e-6) > 2.3 or (w < 1.4 and n_openings >= 2):
        return "connector"  # hallway / corridor
    return "room"


def _edge_meas(e: PlanEdge, prev: PlanEdge, nxt: PlanEdge, tier: str, scale_rel: float | None) -> Measurement:
    L = float(np.linalg.norm(e.b - e.a))
    # a wall's length is bounded by the two perpendicular faces at its ends
    s = length_sigma(L, prev.sigma_face, nxt.sigma_face)
    if scale_rel is not None:
        s = float(np.hypot(s, scale_rel * L))
    snapped = prev.snapped and nxt.snapped
    method = f"{tier}:face-to-face" if snapped else f"{tier}:face-to-freespace"
    return Measurement(L, s, method)


def _area_meas(room: PlanRoom, tier: str) -> Measurement:
    # first-order: dA = sum(edge_len * d_face); faces independent
    var = sum((np.linalg.norm(e.b - e.a) * e.sigma_face) ** 2 for e in room.edges)
    if room.scale_rel is not None:
        var += (2 * room.scale_rel * room.polygon.area) ** 2  # area scales with s^2
    return Measurement(room.polygon.area, float(np.sqrt(var)), f"{tier}:polygon-area-propagated", "m2")


def adjacency(rooms: list[PlanRoom], openings: list[PlanOpening], wall_max: float = 0.35) -> list[dict]:
    out = {}
    for o in openings:
        if len(o.rooms) == 2:
            k = tuple(sorted(o.rooms))
            out.setdefault(k, {"rooms": list(k), "via": [], "shared_wall": False})
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            d = rooms[i].polygon.distance(rooms[j].polygon)
            if d < wall_max:
                inter = rooms[i].polygon.buffer(wall_max / 2 + 0.01).intersection(rooms[j].polygon.buffer(wall_max / 2 + 0.01))
                if inter.area > 0.05:
                    k = (i, j)
                    out.setdefault(k, {"rooms": [i, j], "via": [], "shared_wall": False})
                    out[k]["shared_wall"] = True
    return list(out.values())


def overlaps(rooms: list[PlanRoom], tol_area: float = 0.05) -> list[dict]:
    res = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a = rooms[i].polygon.intersection(rooms[j].polygon).area
            if a > tol_area:
                res.append({"rooms": [i, j], "area_m2": round(a, 3)})
    return res


def to_document(plan: Plan) -> dict:
    k = TIER_K.get(plan.tier, 1.0)
    tier = plan.tier
    rooms_out, openings_out, surfaces = [], [], []
    room_ids = [f"R{i + 1}" for i in range(len(plan.rooms))]
    # assign openings to the nearest edge of each room they touch
    op_ids = [f"O{i + 1}" for i in range(len(plan.openings))]
    op_wall: dict[str, list] = {oid: [] for oid in op_ids}
    for ri, room in enumerate(plan.rooms):
        n_op = sum(1 for o in plan.openings if ri in o.rooms)
        rid = room_ids[ri]
        name = room.name or _classify(room.polygon, n_op)
        walls = []
        height = room.ceiling
        for ei, e in enumerate(room.edges):
            prev = room.edges[ei - 1]
            nxt = room.edges[(ei + 1) % len(room.edges)]
            wid = f"{rid}-W{ei + 1}"
            L = _edge_meas(e, prev, nxt, tier, room.scale_rel)
            walls.append({"id": wid, "start": np.round(e.a, 4).tolist(), "end": np.round(e.b, 4).tolist(),
                          "length": L.to_json(k), "face_sigma_m": round(e.sigma_face * k, 4),
                          "bounded_by_wall_face": bool(e.snapped), "openings": []})
            wall_area = Measurement(L.value * height.value, float(np.hypot(L.sigma * height.value, L.value * height.sigma)),
                                    f"{tier}:length*height", "m2", observed=height.observed) if height else None
            surfaces.append({"id": f"{wid}-S", "room": rid, "kind": "wall", "wall": wid,
                             "area": wall_area.to_json(k) if wall_area else None})
        seg = [LineString([e.a, e.b]) for e in room.edges]
        for oi, o in enumerate(plan.openings):
            if ri not in o.rooms:
                continue
            mid = LineString([o.a, o.b]).centroid
            d = [s.distance(mid) for s in seg]
            ei = int(np.argmin(d))
            if d[ei] < 0.45:
                walls[ei]["openings"].append(op_ids[oi])
                op_wall[op_ids[oi]].append(walls[ei]["id"])
        area = _area_meas(room, tier)
        surfaces.append({"id": f"{rid}-FLOOR", "room": rid, "kind": "floor", "area": area.to_json(k)})
        surfaces.append({"id": f"{rid}-CEIL", "room": rid, "kind": "ceiling", "area": area.to_json(k)})
        perim = Measurement(room.polygon.length, float(np.sqrt(sum(
            (2 * e.sigma_face) ** 2 for e in room.edges)) / np.sqrt(2)), f"{tier}:perimeter", "m")
        rooms_out.append({
            "id": rid, "type": name,
            "polygon": [[round(x, 4), round(y, 4)] for x, y in list(room.polygon.exterior.coords)[:-1]],
            "floor_area": area.to_json(k), "perimeter": perim.to_json(k),
            "ceiling_height": height.to_json(k) if height else None,
            "walls": walls,
        })
    for oi, o in enumerate(plan.openings):
        w = Measurement(float(np.linalg.norm(o.b - o.a)), o.sigma_width, o.method)
        openings_out.append({"id": op_ids[oi], "type": o.kind, "width": w.to_json(k),
                             "start": np.round(o.a, 4).tolist(), "end": np.round(o.b, 4).tolist(),
                             "rooms": [room_ids[r] for r in o.rooms], "walls": op_wall[op_ids[oi]]})
    adj = adjacency(plan.rooms, plan.openings)
    for a in adj:
        a["via"] = [op_ids[i] for i, o in enumerate(plan.openings) if sorted(o.rooms) == a["rooms"]]
        a["rooms"] = [room_ids[r] for r in a["rooms"]]
    union = unary_union([r.polygon for r in plan.rooms]) if plan.rooms else Polygon()
    fp_var = sum(r["floor_area"]["sigma"] ** 2 for r in rooms_out)
    footprint = Measurement(sum(r.polygon.area for r in plan.rooms), float(np.sqrt(fp_var)) / max(k, 1e-9),
                            f"{tier}:sum-room-areas", "m2")
    bounds = union.bounds if not union.is_empty else (0, 0, 0, 0)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "generator": {"name": "scan2plan", "version": __version__},
        "tier": tier,
        "capture": plan.capture,
        "units": {"length": "m", "area": "m2"},
        "frame": {"description": "plan frame: gravity-aligned, x/y on the dominant wall axes, origin arbitrary",
                  "bounds": [round(b, 3) for b in bounds]},
        "property": {
            "n_rooms": len(rooms_out),
            "floor_area_total": footprint.to_json(k),
            "adjacency": adj,
            "overlaps": [{"rooms": [room_ids[i] for i in o["rooms"]], "area_m2": o["area_m2"]} for o in overlaps(plan.rooms)],
            "drift_correction": plan.drift,
        },
        "rooms": rooms_out,
        "openings": openings_out,
        "surfaces": surfaces,
        "damage": plan.extra.get("damage", []),
        "concealed_damage_flags": plan.extra.get("flags", []),
        "scope": plan.extra.get("scope", []),
        "calibration": {"tier_sigma_multiplier": k},
        "timing_s": {kk: round(v, 2) for kk, v in plan.timing.items()},
        "notes": plan.notes,
    }
    return doc
