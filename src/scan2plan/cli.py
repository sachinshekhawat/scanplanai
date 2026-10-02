"""One command per capture:

    scan2plan run <capture> [--tier auto|lidar|video|photo] [--out runs/<name>] [--no-drift]

<capture> may be a Stray Scanner folder / zip (lidar or video tier) or a folder of per-room
photo folders (photo tier). Writes plan.json (validated), plan.png, plan.svg, log.txt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path


def _unzip_if_needed(p: Path) -> Path:
    if p.suffix.lower() != ".zip":
        return p
    dest = Path("data/raw") / p.stem
    if not dest.exists():
        dest.mkdir(parents=True)
        with zipfile.ZipFile(p) as z:
            z.extractall(dest)
    return dest


def detect_tier(p: Path) -> str:
    if list(p.rglob("odometry.csv")):
        has_depth = any(d.is_dir() and any(d.iterdir()) for d in p.rglob("depth"))
        return "lidar" if has_depth else "video"
    if list(p.rglob("*.mp4")) or list(p.rglob("*.mov")) or list(p.rglob("*.MOV")):
        return "video"
    return "photo"


def run(capture: str, tier: str = "auto", out: str | None = None, drift: bool = True) -> dict:
    from scan2plan.damage import attach_damage
    from scan2plan.plan import to_document
    from scan2plan.render import render
    from scan2plan.schema import validate

    src = _unzip_if_needed(Path(capture))
    tier = detect_tier(src) if tier == "auto" else tier
    name = Path(capture).stem
    outdir = Path(out or f"runs/{name}_{tier}{'' if drift else '_nodrift'}")
    outdir.mkdir(parents=True, exist_ok=True)
    logf = open(outdir / "log.txt", "w")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    t0 = time.time()
    log(f"capture={src} tier={tier} drift={drift}")
    meta = {"name": name, "source": str(src), "tier": tier}
    if tier == "lidar":
        from scan2plan.lidar.pipeline import run_lidar
        from scan2plan.lidar.to_plan import lidar_to_plan
        res = run_lidar(src, drift=drift, log=log)
        meta.update({"n_frames": int(res.cap.n_frames),
                     "duration_s": round(float(res.cap.timestamps[-1] - res.cap.timestamps[0]), 1),
                     "device_class": "iPhone/iPad Pro with LiDAR (Stray Scanner)"})
        plan = lidar_to_plan(res, meta)
        ctx = {"lidar": res}
    elif tier == "video":
        from scan2plan.video.pipeline import run_video
        plan, ctx = run_video(src, meta, drift=drift, log=log)
    elif tier == "photo":
        from scan2plan.photo.pipeline import run_photo
        plan, ctx = run_photo(src, meta, log=log)
    else:
        raise SystemExit(f"unknown tier {tier}")
    t1 = time.time()
    attach_damage(plan, ctx, log=log)
    plan.timing["damage_s"] = time.time() - t1
    plan.timing["wall_clock_s"] = time.time() - t0
    doc = to_document(plan)
    validate(doc)
    (outdir / "plan.json").write_text(json.dumps(doc, indent=1))
    render(doc, outdir / "plan")
    log(f"wrote {outdir/'plan.json'} ({doc['property']['n_rooms']} rooms, {len(doc['openings'])} openings, "
        f"{len(doc['damage'])} damage regions) in {time.time() - t0:.1f} s")
    logf.close()
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(prog="scan2plan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process one capture")
    r.add_argument("capture")
    r.add_argument("--tier", default="auto", choices=["auto", "lidar", "video", "photo"])
    r.add_argument("--out")
    r.add_argument("--no-drift", action="store_true", help="ablation: use device poses as-is")
    v = sub.add_parser("validate", help="validate a plan.json against the schema")
    v.add_argument("json")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.capture, a.tier, a.out, drift=not a.no_drift)
    elif a.cmd == "validate":
        from scan2plan.schema import validate
        validate(json.loads(Path(a.json).read_text()))
        print("valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
