"""Render a plan document (JSON) to PNG and SVG: rooms, dimensions with intervals, openings, damage."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon as MplPoly  # noqa: E402

PALETTE = ["#cfe8fc", "#fde2c8", "#d7f5d0", "#f3d4f5", "#fff4bf", "#d4f1f1", "#f8d0d0", "#e2e2f8"]
DAMAGE_COLORS = {"water_stain": "#1f77b4", "mold": "#2ca02c", "crack": "#d62728", "hole": "#9467bd",
                 "peeling_paint": "#ff7f0e", "other": "#7f7f7f"}


def _fmt(m: dict, cm: bool = False) -> str:
    if m is None:
        return "?"
    v, s = m["value"], m["sigma"] * 1.96
    if cm:
        return f"{v * 100:.0f}±{s * 100:.0f} cm"
    return f"{v:.2f}±{s:.2f}" + ("" if m.get("observed", True) else "*")


def render(doc: dict, out_base: str | Path, title: str | None = None) -> list[Path]:
    out_base = Path(out_base)
    rooms = doc["rooms"]
    fig, ax = plt.subplots(figsize=(13, 13))
    allp = np.concatenate([np.array(r["polygon"]) for r in rooms]) if rooms else np.zeros((1, 2))
    span = float(np.ptp(allp, 0).max()) if len(allp) > 1 else 1
    fs = max(6, min(10, 90 / max(span, 1)))
    for i, r in enumerate(rooms):
        P = np.array(r["polygon"])
        ax.add_patch(MplPoly(P, closed=True, fc=PALETTE[i % len(PALETTE)], ec="#222", lw=2.2, zorder=1))
        c = P.mean(0)
        try:
            from shapely.geometry import Polygon
            c = np.array(Polygon(P).representative_point().coords[0])
        except Exception:
            pass
        ch = r.get("ceiling_height")
        ax.text(c[0], c[1], f"{r['id']} ({r['type']})\n{_fmt(r['floor_area'])} m²\nH {_fmt(ch)} m",
                ha="center", va="center", fontsize=fs + 1, zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
        for w in r["walls"]:
            a, b = np.array(w["start"]), np.array(w["end"])
            L = w["length"]["value"]
            if L < 0.35:
                continue
            m = (a + b) / 2
            d = (b - a) / max(L, 1e-9)
            nrm = np.array([-d[1], d[0]])
            # put label inside the room
            if not MplPoly(P).get_path().contains_point(m + nrm * 0.05):
                nrm = -nrm
            p = m + nrm * 0.18
            ang = np.degrees(np.arctan2(d[1], d[0]))
            if ang > 90 or ang < -90:
                ang += 180
            ax.text(p[0], p[1], _fmt(w["length"]), rotation=ang, ha="center", va="center",
                    fontsize=fs - 1, color="#333", zorder=4)
    for o in doc["openings"]:
        a, b = np.array(o["start"]), np.array(o["end"])
        col = "#2a9d2a" if o["type"] == "door" else ("#1e6fd9" if o["type"] == "window" else "#e09f00")
        ax.plot([a[0], b[0]], [a[1], b[1]], color="white", lw=6, zorder=6, solid_capstyle="butt")
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=3, zorder=7, solid_capstyle="butt")
        m = (a + b) / 2
        ax.text(m[0], m[1], f"{o['id']} {_fmt(o['width'], cm=True)}", fontsize=fs - 1, color=col, zorder=8,
                ha="center", va="bottom", bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))
    for dmg in doc.get("damage", []):
        fp = dmg.get("plan_footprint")
        if fp:
            P = np.array(fp)
            ax.add_patch(MplPoly(P, closed=True, fc=DAMAGE_COLORS.get(dmg["class"], "#7f7f7f"), alpha=0.6, zorder=9))
            ax.text(*P.mean(0), dmg["id"], fontsize=fs - 1, zorder=10)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.05)
    ax.set_xlabel("m")
    ax.set_ylabel("m")
    ax.grid(alpha=0.2)
    fa = doc["property"]["floor_area_total"]
    ax.set_title(title or f"{doc['capture'].get('name', '')} — tier {doc['tier']} — {doc['property']['n_rooms']} rooms, "
                 f"{_fmt(fa)} m² (95% CI; * = not observed)", fontsize=12)
    outs = []
    for ext in ("png", "svg"):
        p = out_base.with_suffix("." + ext)
        fig.savefig(p, dpi=110 if ext == "png" else None, bbox_inches="tight")
        outs.append(p)
    plt.close(fig)
    return outs
