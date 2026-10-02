"""Benchmark harness: regenerates every reported number from raw inputs.

    python -m scan2plan.benchmark.run [--tiers lidar,video,photo] [--tag NAME] [--skip-run]

Writes benchmark/results/<tag>/{metrics.json, report.md} plus per-capture runs under
runs/bench_<tag>/. Gates follow the case-study table. Reference used for each metric:
  * ground truth: benchmark/ground_truth.csv if present (tape/laser);
  * otherwise the LiDAR tier of the same capture is the reference for video/photo tiers
    (disclosed as "LiDAR-referenced", NOT ground truth), and LiDAR accuracy gates are
    reported as "no GT" with only repeatability/self-consistency evidence.
Repeatability pair: single_scan_floor_only vs single_scan_with_ceiling (same apartment,
two sessions, same tier).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[3]
CAPTURES = ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]
REPEAT_PAIR = ("single_scan_with_ceiling", "single_scan_floor_only")
Z95 = 1.96


# ------------------------------------------------------------------ helpers
def _poly(r):
    return Polygon(r["polygon"])


def _tf_doc(doc, k, t):
    """Rotate a document's rooms/openings by k*90 deg and translate (evaluation only)."""
    R = np.linalg.matrix_power(np.array([[0, -1], [1, 0]]), k % 4)
    out = json.loads(json.dumps(doc))
    for r in out["rooms"]:
        r["polygon"] = (np.array(r["polygon"]) @ R.T + t).tolist()
        for w in r["walls"]:
            w["start"] = (np.array(w["start"]) @ R.T + t).tolist()
            w["end"] = (np.array(w["end"]) @ R.T + t).tolist()
    for o in out["openings"]:
        o["start"] = (np.array(o["start"]) @ R.T + t).tolist()
        o["end"] = (np.array(o["end"]) @ R.T + t).tolist()
    return out


def align_docs(ref, doc, scale_free=False):
    """Best k*90 rotation + translation (+ optional scale) aligning doc's rooms onto ref's, by
    maximising union IoU over a coarse-to-fine translation search on room-centroid pairs."""
    A = [_poly(r) for r in ref["rooms"]]
    from shapely.ops import unary_union

    UA = unary_union(A)
    best = (-1, 0, np.zeros(2))
    for k in range(4):
        B = [affinity.affine_transform(_poly(r), list(np.linalg.matrix_power(np.array([[0, -1], [1, 0]]), k).ravel()) + [0, 0])
             for r in doc["rooms"]]
        UB = unary_union(B)
        if UB.is_empty:
            continue
        cands = [np.array([a.centroid.x - b.centroid.x, a.centroid.y - b.centroid.y]) for a in A for b in B]
        cands.append(np.array([UA.centroid.x - UB.centroid.x, UA.centroid.y - UB.centroid.y]))
        for t in cands:
            for step in (0.2, 0.05, 0.01):
                improved = True
                while improved:
                    improved = False
                    for d in ([step, 0], [-step, 0], [0, step], [0, -step]):
                        tt = t + d
                        iou = UA.intersection(affinity.translate(UB, *tt)).area / max(UA.union(affinity.translate(UB, *tt)).area, 1e-9)
                        cur = UA.intersection(affinity.translate(UB, *t)).area / max(UA.union(affinity.translate(UB, *t)).area, 1e-9)
                        if iou > cur + 1e-6:
                            t, improved = tt, True
            iou = UA.intersection(affinity.translate(UB, *t)).area / max(UA.union(affinity.translate(UB, *t)).area, 1e-9)
            if iou > best[0]:
                best = (iou, k, t)
    return best


def match_rooms(ref, doc, min_iou=0.3):
    pairs = []
    for i, ra in enumerate(ref["rooms"]):
        pa = _poly(ra)
        best = None
        for j, rb in enumerate(doc["rooms"]):
            pb = _poly(rb)
            iou = pa.intersection(pb).area / max(pa.union(pb).area, 1e-9)
            if best is None or iou > best[0]:
                best = (iou, j)
        if best and best[0] >= min_iou:
            pairs.append((i, best[1], best[0]))
    return pairs


def match_walls(ra, rb, tol=0.35):
    """Pair walls of two matched rooms by orientation and position of their midpoints."""
    out = []
    for wa in ra["walls"]:
        a0, a1 = np.array(wa["start"]), np.array(wa["end"])
        da = a1 - a0
        ma = (a0 + a1) / 2
        best = None
        for wb in rb["walls"]:
            b0, b1 = np.array(wb["start"]), np.array(wb["end"])
            db = b1 - b0
            if abs(abs(np.dot(da, db)) / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-9) - 1) > 0.05:
                continue
            d = np.linalg.norm((b0 + b1) / 2 - ma)
            if d < tol and (best is None or d < best[0]):
                best = (d, wb)
        if best is not None and wa["length"]["value"] > 0.5:
            out.append((wa, best[1]))
    return out


def match_openings(ref, doc, tol=0.4):
    pairs, used = [], set()
    for oa in ref["openings"]:
        ma = (np.array(oa["start"]) + np.array(oa["end"])) / 2
        best = None
        for j, ob in enumerate(doc["openings"]):
            if j in used:
                continue
            d = np.linalg.norm((np.array(ob["start"]) + np.array(ob["end"])) / 2 - ma)
            if d < tol and (best is None or d < best[0]):
                best = (d, j)
        if best:
            used.add(best[1])
            pairs.append((oa, doc["openings"][best[1]]))
    return pairs, len(ref["openings"]) - len(pairs), len(doc["openings"]) - len(used)


def coverage(errs, sigmas):
    """Fraction of |error| inside the stated 95% interval, and mean |z|."""
    e, s = np.abs(np.array(errs)), np.maximum(np.array(sigmas), 1e-6)
    if len(e) == 0:
        return None
    return {"n": int(len(e)), "coverage95": float(np.mean(e <= Z95 * s)), "median_abs_z": float(np.median(e / s))}


def load_gt():
    f = ROOT / "benchmark" / "ground_truth.csv"
    if not f.exists():
        return None
    rows = [r for r in csv.DictReader(l for l in open(f) if not l.startswith("#"))]
    return rows


# ------------------------------------------------------------------ runs
def run_all(tiers, tag, skip_run=False, log=print):
    from scan2plan.cli import run

    out = {}
    timing = {}
    for cap in CAPTURES:
        for tier in tiers:
            src = ROOT / "data" / "raw" / cap
            if tier == "photo":
                src = ROOT / "data" / "photo_sets" / cap
                if not src.exists():
                    import subprocess
                    import sys

                    subprocess.run([sys.executable, str(ROOT / "scripts/make_photo_set.py"), str(ROOT / "data/raw" / cap),
                                    str(src), "6"], check=True, capture_output=True)
            for drift in ([True, False] if tier == "lidar" else [True]):
                name = f"{cap}_{tier}{'' if drift else '_nodrift'}"
                d = ROOT / "runs" / f"bench_{tag}" / name
                f = d / "plan.json"
                if not (skip_run and f.exists()):
                    t0 = time.time()
                    try:
                        run(str(src), tier, str(d), drift=drift)
                    except Exception as e:  # a failing tier is a reported result, not a crash
                        log(f"[bench] {name} FAILED: {e!r}")
                        out[name] = {"error": repr(e)}
                        continue
                    timing[name] = time.time() - t0
                doc = json.loads(f.read_text())
                out[name] = doc
                timing.setdefault(name, doc["timing_s"].get("wall_clock_s"))
    return out, timing


# ------------------------------------------------------------------ metrics
def metrics(docs, timing):
    M = {"reference_note": "No tape/laser ground truth in the provided sample data. LiDAR is the reference "
                           "for video/photo (LiDAR-referenced error, not GT). LiDAR absolute accuracy gates "
                           "need benchmark/ground_truth.csv.", "gates": {}}
    gt = load_gt()
    M["ground_truth_rows"] = len(gt) if gt else 0

    # 1. repeatability: two sessions of the same apartment, same tier
    a, b = docs.get(f"{REPEAT_PAIR[0]}_lidar"), docs.get(f"{REPEAT_PAIR[1]}_lidar")
    rep = {}
    if a and b and "rooms" in a and "rooms" in b:
        iou, k, t = align_docs(a, b)
        bb = _tf_doc(b, k, t)
        rows = []
        for i, j, riou in match_rooms(a, bb):
            for wa, wb in match_walls(a["rooms"][i], bb["rooms"][j]):
                L = wa["length"]["value"]
                d = wb["length"]["value"] - L
                rows.append({"room_a": a["rooms"][i]["id"], "room_b": bb["rooms"][j]["id"], "wall_a": wa["id"],
                             "len_a": L, "len_b": wb["length"]["value"], "diff_m": d,
                             "pass": abs(d) <= max(0.01, 0.005 * L),
                             "sigma_diff": float(np.hypot(wa["length"]["sigma"], wb["length"]["sigma"]))})
        rep = {"footprint_iou_after_alignment": iou, "n_room_pairs": len(match_rooms(a, bb)),
               "n_rooms": [len(a["rooms"]), len(b["rooms"])], "walls": rows,
               "pass_rate": float(np.mean([r["pass"] for r in rows])) if rows else None,
               "median_abs_diff_m": float(np.median([abs(r["diff_m"]) for r in rows])) if rows else None,
               "calibration": coverage([r["diff_m"] for r in rows], [r["sigma_diff"] for r in rows])}
        # ceiling: only with_ceiling observes it -> spread across captures not measurable
        M["gates"]["repeatability"] = {"gate": "|diff| <= max(1 cm, 0.5%) per wall", "value": rep["pass_rate"],
                                       "pass": bool(rep["pass_rate"] is not None and rep["pass_rate"] >= 0.95)}
    M["repeatability"] = rep

    # 2. drift ablation: stitched footprint with correction on vs off
    abl = {}
    for cap in ("single_scan_floor_only", "single_scan_with_ceiling"):
        on, off = docs.get(f"{cap}_lidar"), docs.get(f"{cap}_lidar_nodrift")
        if on and off and "rooms" in on and "rooms" in off:
            abl[cap] = {"on": {"n_rooms": on["property"]["n_rooms"], "area_m2": on["property"]["floor_area_total"]["value"],
                               "overlaps": len(on["property"]["overlaps"]), "drift": on["property"]["drift_correction"]},
                        "off": {"n_rooms": off["property"]["n_rooms"], "area_m2": off["property"]["floor_area_total"]["value"],
                                "overlaps": len(off["property"]["overlaps"])}}
    if a and b:
        # cross-session agreement is the drift-sensitive number: same flat, two walks
        aoff, boff = docs.get(f"{REPEAT_PAIR[0]}_lidar_nodrift"), docs.get(f"{REPEAT_PAIR[1]}_lidar_nodrift")
        if aoff and boff:
            abl["cross_session_footprint_iou"] = {"on": align_docs(a, b)[0], "off": align_docs(aoff, boff)[0]}
    M["drift_ablation"] = abl
    M["gates"]["drift_accountability"] = {"gate": "drift handling stated + on/off ablation", "pass": bool(abl),
                                          "value": abl.get("cross_session_footprint_iou")}

    # 3. ceiling height
    ch = {}
    for name, d in docs.items():
        if "rooms" not in d or not name.endswith("_lidar"):
            continue
        ch[name] = [{"room": r["id"], "h": r["ceiling_height"]["value"], "sigma": r["ceiling_height"]["sigma"],
                     "observed": r["ceiling_height"]["observed"]} for r in d["rooms"] if r["ceiling_height"]]
    obs = [x for v in ch.values() for x in v if x["observed"]]
    M["ceiling"] = {"per_capture": ch, "n_observed_rooms": len(obs),
                    "repeat_spread_m": None,
                    "note": "Only single_scan_with_ceiling observes the ceiling; the other session did not scan it, "
                            "so cross-capture spread is not measurable on sample data (reported as unobserved prior)."}
    M["gates"]["ceiling_height"] = {"gate": "<= 1.5 cm vs GT; spread <= 1 cm", "pass": None, "value": "no GT"}

    # 4. cross-tier: video and photo vs LiDAR reference (same capture)
    xt = {}
    for cap in CAPTURES:
        ref = docs.get(f"{cap}_lidar")
        if not ref or "rooms" not in ref:
            continue
        for tier, gate in (("video", 0.03), ("photo", 0.08)):
            d = docs.get(f"{cap}_{tier}")
            if not d or "rooms" not in d or not d["rooms"]:
                xt[f"{cap}_{tier}"] = {"error": d.get("error") if d else "missing"}
                continue
            iou, k, t = align_docs(ref, d)
            dd = _tf_doc(d, k, t)
            rows = []
            for i, j, riou in match_rooms(ref, dd, 0.2):
                for wa, wb in match_walls(ref["rooms"][i], dd["rooms"][j], tol=0.6):
                    L = wa["length"]["value"]
                    e = wb["length"]["value"] - L
                    rows.append({"L_ref": L, "err": e, "rel": e / L, "sigma": wb["length"]["sigma"], "pass": abs(e / L) <= gate})
            fa_ref = ref["property"]["floor_area_total"]["value"]
            fa = d["property"]["floor_area_total"]
            xt[f"{cap}_{tier}"] = {
                "footprint_iou": iou, "n_rooms": [len(ref["rooms"]), len(d["rooms"])],
                "floor_area_rel_err": (fa["value"] - fa_ref) / fa_ref,
                "floor_area_in_ci": bool(fa["ci95"][0] <= fa_ref <= fa["ci95"][1]),
                "n_walls_matched": len(rows),
                "wall_pass_rate": float(np.mean([r["pass"] for r in rows])) if rows else None,
                "wall_median_abs_rel": float(np.median([abs(r["rel"]) for r in rows])) if rows else None,
                "calibration": coverage([r["err"] for r in rows], [r["sigma"] for r in rows]),
                "overlaps": len(d["property"]["overlaps"]),
            }
    M["cross_tier"] = xt
    for tier, gate in (("video", "walls within +-3% (LiDAR-referenced)"), ("photo", "walls within +-8%, footprint +-8%, no overlaps")):
        vals = [v for k, v in xt.items() if k.endswith(tier) and "wall_pass_rate" in v and v["wall_pass_rate"] is not None]
        pr = float(np.mean([v["wall_pass_rate"] for v in vals])) if vals else None
        M["gates"][f"{tier}_tier"] = {"gate": gate, "value": pr, "pass": bool(pr is not None and pr >= 0.85)}

    # 5. openings (LiDAR self-consistency across sessions)
    if a and b:
        iou, k, t = align_docs(a, b)
        pairs, miss, phantom = match_openings(a, _tf_doc(b, k, t))
        diffs = [ob["width"]["value"] - oa["width"]["value"] for oa, ob in pairs]
        M["openings_repeatability"] = {"matched": len(pairs), "missed": miss, "phantom": phantom,
                                       "within_2cm": int(np.sum(np.abs(diffs) <= 0.02)) if diffs else 0,
                                       "diffs_m": diffs}
        n_tot = len(pairs) + miss + phantom
        M["gates"]["opening_widths"] = {"gate": "<= 2 cm on >= 85% (miss & phantom count as fail); cross-session proxy",
                                        "value": (int(np.sum(np.abs(diffs) <= 0.02)) / n_tot) if n_tot else None,
                                        "pass": bool(n_tot and np.sum(np.abs(diffs) <= 0.02) / n_tot >= 0.85)}
    # 6. photo stitch gate (overlaps, adjacency)
    ph = {k: v for k, v in docs.items() if k.endswith("_photo") and "rooms" in v}
    M["photo_stitch"] = {k: {"n_rooms": v["property"]["n_rooms"], "overlaps": v["property"]["overlaps"],
                             "adjacency": len(v["property"]["adjacency"]), "notes": v.get("notes", [])} for k, v in ph.items()}
    # 7. damage + timing
    M["damage"] = {k: {"n_regions": len(v["damage"]), "classes": sorted({d["class"] for d in v["damage"]}),
                       "flags": [f["rule"] for f in v["concealed_damage_flags"]]} for k, v in docs.items() if "damage" in v}
    M["timing_s"] = timing
    return M


def report_md(M) -> str:
    L = ["# Benchmark report (auto-generated)", "", M["reference_note"], "", "## Gates", "",
         "| gate | criterion | value | pass |", "|---|---|---|---|"]
    for k, g in M["gates"].items():
        v = g.get("value")
        vs = json.dumps(v) if not isinstance(v, float) else f"{v:.3f}"
        L.append(f"| {k} | {g['gate']} | {vs[:80]} | {g.get('pass')} |")
    rep = M.get("repeatability") or {}
    if rep:
        L += ["", "## Repeatability (two sessions, LiDAR tier)", "",
              f"Footprint IoU after alignment {rep['footprint_iou_after_alignment']:.3f}; rooms {rep['n_rooms']}; "
              f"matched room pairs {rep['n_room_pairs']}; wall pass rate {rep['pass_rate']}; median |diff| "
              f"{rep['median_abs_diff_m']}; calibration {rep['calibration']}", "",
              "| room A | wall | len A (m) | len B (m) | diff (cm) | pass |", "|---|---|---|---|---|---|"]
        for r in rep["walls"]:
            L.append(f"| {r['room_a']} | {r['wall_a']} | {r['len_a']:.3f} | {r['len_b']:.3f} | {r['diff_m'] * 100:+.1f} | {r['pass']} |")
    L += ["", "## Drift ablation", "", "```", json.dumps(M["drift_ablation"], indent=1, default=float)[:3000], "```",
          "", "## Cross-tier (LiDAR-referenced)", "", "| capture/tier | footprint IoU | rooms ref/tier | area rel err | area in CI | walls | wall pass | median rel | calib cov95 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, v in M["cross_tier"].items():
        if "error" in v:
            L.append(f"| {k} | FAILED: {v['error'][:60]} | | | | | | | |")
            continue
        c = v["calibration"] or {}
        L.append(f"| {k} | {v['footprint_iou']:.2f} | {v['n_rooms']} | {v['floor_area_rel_err']:+.1%} | {v['floor_area_in_ci']} | "
                 f"{v['n_walls_matched']} | {v['wall_pass_rate']} | {v['wall_median_abs_rel']} | {c.get('coverage95')} |")
    L += ["", "## Openings (cross-session)", "", "```", json.dumps(M.get("openings_repeatability"), default=float), "```",
          "", "## Ceiling", "", M["ceiling"]["note"], "", "## Damage", "", "```", json.dumps(M["damage"], indent=1), "```",
          "", "## Timing (s, wall clock per capture)", "", "| run | s |", "|---|---|"]
    for k, v in M["timing_s"].items():
        L.append(f"| {k} | {v:.0f} |" if v else f"| {k} | ? |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="lidar,video,photo")
    ap.add_argument("--tag", default="current")
    ap.add_argument("--skip-run", action="store_true", help="reuse existing runs/bench_<tag> outputs")
    a = ap.parse_args(argv)
    docs, timing = run_all(a.tiers.split(","), a.tag, a.skip_run)
    M = metrics(docs, timing)
    out = ROOT / "benchmark" / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(M, indent=1, default=float))
    (out / "report.md").write_text(report_md(M))
    print(report_md(M))


if __name__ == "__main__":
    main()
