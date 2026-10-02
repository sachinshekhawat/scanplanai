"""Damage regions -> concealed-damage flags (explicit rules) -> scope line items keyed to surfaces."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from scan2plan.measure import Measurement

# ---------------------------------------------------------------- concealed-damage rules
# Each rule: id, predicate over (region, context), severity, rationale. Rules are deliberately
# simple, auditable and printed verbatim into the flag.
RULES = [
    ("R1_ceiling_water", lambda d, c: d["class"] in ("water_stain", "mold") and d["surface_kind"] == "ceiling",
     "high", "Water staining/mould on a ceiling indicates a leak above (roof, plumbing, upstairs wet room); "
             "the cavity above is likely wet. Inspect and moisture-test the void before repair."),
    ("R2_low_wall_moisture", lambda d, c: d["class"] in ("water_stain", "mold", "peeling_paint")
     and d["surface_kind"] == "wall" and d["uv_box"][1] < 0.5,
     "high", "Moisture damage starting within 0.5 m of the floor suggests rising damp or a leak behind/below the "
             "wall; insulation, studs and skirting behind the surface may be wet. Moisture-meter the cavity."),
    ("R3_mold_extent", lambda d, c: d["class"] == "mold" and d["area"][0] > 0.2,
     "high", "Visible mould > 0.2 m2 usually means hidden growth behind the surface (cavity side). "
             "Remediate with containment; open up the wall if moisture readings are high."),
    ("R4_wet_room", lambda d, c: d["class"] in ("water_stain", "mold", "peeling_paint") and c["room_type"].get(d["room"]) == "closet_or_wc",
     "medium", "Moisture damage in a small wet room (bath/WC) points to failed sealant, grout or a pipe leak; "
               "subfloor and the adjacent wall cavity may be affected."),
    ("R5_long_crack", lambda d, c: d["class"] == "crack" and d["length"][0] > 0.5,
     "medium", "A crack longer than 0.5 m can reflect movement of the structure or a failed board joint; check "
               "for continuation behind finishes and on the other side of the wall."),
    ("R6_crack_at_opening", lambda d, c: d["class"] == "crack" and d["surface_kind"] == "wall" and c["near_opening"](d),
     "medium", "Cracks running from the corner of a door/window opening are typical of lintel deflection or "
               "settlement; inspect the lintel and structure above the opening."),
]

# ---------------------------------------------------------------- scope (repair) line items
UNIT_RATES_NOTE = "quantities only; pricing intentionally omitted (use local rate tables)"


def _scope_for(d: dict, surf: dict) -> list[dict]:
    area = surf.get("area") or {}
    items = []
    A = d["area"]
    if d["class"] == "crack":
        L = d["length"]
        items.append(("rake out and fill crack, sand flush", Measurement(L[0] * 1.1, L[1] * 1.1, "damage:length+10%"), "m"))
    elif d["class"] == "hole":
        items.append(("patch hole (mesh + filler / board patch)", Measurement(max(A[0], 0.01) * 1.5, A[1] * 1.5, "damage:area+50%"), "m2"))
    elif d["class"] == "mold":
        items.append(("mould remediation incl. 0.3 m margin, biocide wash",
                      Measurement((np.sqrt(A[0]) + 0.6) ** 2, A[1] * 2, "damage:area+0.3m margin"), "m2"))
    elif d["class"] in ("water_stain", "peeling_paint"):
        items.append(("scrape loose material, stain-block primer on affected area",
                      Measurement(A[0] * 1.3, A[1] * 1.3, "damage:area+30%"), "m2"))
    # finish: the whole surface is redecorated so the repair does not show
    if area and area.get("value"):
        items.append(("redecorate full surface (mist coat + 2 coats)",
                      Measurement(area["value"], area["sigma"], "surface:area"), "m2"))
    return items


def _views_lidar(res, max_views: int = 260, log=print):
    """Keyframes with RGB synced to the pose/depth stream (video lags odometry, see io/sync)."""
    from scan2plan.damage.detect import View
    from scan2plan.io.sync import estimate_offsets, video_to_pose_frame
    from scan2plan.lidar.fusion import select_keyframes

    cap = res.cap
    Rs = res.drift.rotations if res.drift is not None else cap.rotations
    ts = res.drift.positions if res.drift is not None else cap.positions
    t0 = time.time()
    knots, offs = estimate_offsets(cap, step=60)
    log(f"[damage] rgb/odometry offset {list(map(int, offs))} frames ({time.time() - t0:.0f} s)")
    kf = select_keyframes(cap, min_trans=0.3, min_rot_deg=12, max_gap=90)
    if len(kf) > max_views:
        kf = kf[np.linspace(0, len(kf) - 1, max_views).astype(int)]
    vc = cv2.VideoCapture(str(cap.root / "rgb.mp4"))
    n_vid = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    # pose frame p matches video frame v where p = v + off(v)  =>  v ~ p - off
    vid_for_pose = {}
    vv = np.arange(n_vid)
    pv = video_to_pose_frame(knots, offs, vv)
    for v, p in zip(vv, pv):
        vid_for_pose.setdefault(int(p), int(v))
    views = []
    for i in kf:
        v = vid_for_pose.get(int(i))
        if v is None:
            continue
        vc.set(cv2.CAP_PROP_POS_FRAMES, v)
        ok, f = vc.read()
        if not ok:
            continue
        d = cap.depth(i)
        d[cap.confidence(i) < 2] = 0
        views.append(View(f"f{i}", f, d, cap.K_depth, Rs[i], ts[i], orient=1))
    return views


def _views_video(res, ctx, max_views: int = 120):
    from scan2plan.damage.detect import View

    pc = res.cap
    fdir = ctx.get("frames_dir")
    if fdir is None:
        return []
    idx = np.linspace(0, pc.n_frames - 1, min(max_views, pc.n_frames)).astype(int)
    out = []
    for i in idx:
        img = cv2.imread(str(Path(fdir) / f"{pc.names[i]}.jpg"))
        if img is None:
            continue
        d = pc.depths[i].copy()
        d[pc.confs[i] < 2] = 0
        out.append(View(f"v{pc.names[i]}", img, d, pc.K_depth, pc.rotations[i], pc.positions[i], orient=ctx.get("orient", 0)))
    return out


def attach_damage(plan, ctx, log=print) -> None:
    from scan2plan.damage.detect import detect, surfaces_from_doc
    from scan2plan.plan import to_document

    plan.extra.setdefault("damage", [])
    plan.extra.setdefault("flags", [])
    plan.extra.setdefault("scope", [])
    res = ctx.get("lidar")
    if res is None:
        plan.notes.append("Damage detection not run for this tier input (no posed views available).")
        return
    doc = to_document(plan)
    surfs = surfaces_from_doc(doc, res.R2, res.level.floor_y)
    try:
        views = _views_video(res, ctx) if ctx.get("video") else _views_lidar(res, log=log)
        work = Path("data/cache/damage") / (doc["capture"].get("name", "capture") + "_" + plan.tier)
        regions = detect(views, surfs, res.R2, work, log=log)
    except Exception as e:  # damage must never take the plan down
        log(f"[damage] skipped: {e!r}")
        plan.notes.append(f"Damage detection failed: {e!r}")
        return
    surf_doc = {s["id"]: s for s in doc["surfaces"]}
    room_type = {r["id"]: r["type"] for r in doc["rooms"]}
    opening_pts = [(np.array(o["start"]), np.array(o["end"]), o["walls"]) for o in doc["openings"]]

    def near_opening(d):
        wall_id = d["surface"][:-2]
        return any(wall_id in ws for _, _, ws in opening_pts)

    ctx_r = {"room_type": room_type, "near_opening": near_opening}
    dmg, flags, scope = [], [], []
    for k, d in enumerate(regions, start=1):
        did = f"D{k}"
        s = d["surface_obj"]
        fp = None
        if s.kind == "wall":
            u0, _, u1, _ = d["uv_box"]
            a = np.array([s.c, s.lo + u0]) if s.axis == 0 else np.array([s.lo + u0, s.c])
            b = np.array([s.c, s.lo + u1]) if s.axis == 0 else np.array([s.lo + u1, s.c])
            n = np.array([0.08, 0]) if s.axis == 0 else np.array([0, 0.08])
            fp = [a - n, b - n, b + n, a + n]
        else:
            u0, v0, u1, v1 = d["uv_box"]
            fp = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]]
        rec = {"id": did, "surface": d["surface"], "class": d["class"], "confidence": round(d["confidence"], 3),
               "n_views": d["n_views"],
               "extent": {"area": Measurement(*d["area"], "damage:surface-cells", "m2").to_json(),
                          "width": Measurement(*d["width"], "damage:surface-u", "m").to_json(),
                          "height": Measurement(*d["height"], "damage:surface-v", "m").to_json()},
               "position_on_surface_m": {"u0": round(d["uv_box"][0], 3), "v0": round(d["uv_box"][1], 3),
                                         "u1": round(d["uv_box"][2], 3), "v1": round(d["uv_box"][3], 3)},
               "plan_footprint": [list(map(float, p)) for p in fp]}
        dmg.append(rec)
        for rid, pred, sev, why in RULES:
            if pred(d, ctx_r):
                flags.append({"id": f"F{len(flags) + 1}", "surface": d["surface"], "rule": rid, "severity": sev,
                              "evidence": [did], "rationale": why})
        for action, q, unit in _scope_for(d, surf_doc.get(d["surface"], {})):
            scope.append({"id": f"S{len(scope) + 1}", "surface": d["surface"], "action": action,
                          "quantity": q.to_json(), "unit": unit, "because": [did]})
    for f in flags:
        scope.append({"id": f"S{len(scope) + 1}", "surface": f["surface"],
                      "action": f"investigate concealed damage ({f['rule']}): moisture/structural survey",
                      "quantity": Measurement(1, 0, "count", "count").to_json(), "unit": "each",
                      "because": [f["id"]]})
    plan.extra["damage"], plan.extra["flags"], plan.extra["scope"] = dmg, flags, scope
    plan.notes.append("Damage: classical candidates + CLIP ViT-B/32 zero-shot classification (no damage-specific "
                      "training); confidence = CLIP probability. Scope: " + UNIT_RATES_NOTE + ".")
