# scan2plan

You give it a phone capture: photos, a video, or a LiDAR scan. It returns a dimensioned, stitched floor plan as JSON (schema `schema/plan.schema.json`) plus PNG and SVG renders. Every number in the output carries a 95% interval. The output also lists damage regions, concealed-damage flags and repair scope line items. It runs as **one command per capture** and works offline once model weights are fetched.

| Tier | Input | Device | Core method |
|---|---|---|---|
| LiDAR | Stray Scanner folder or zip (depth + poses + intrinsics) | iPhone/iPad Pro | Depth fusion, wall-face extraction, plan-view ray carving, drift-corrected submap pose graph |
| Video | Any `.mp4`/`.MOV` walkthrough (Stray `rgb.mp4` also works) | Any iPhone 15+ | COLMAP fragments, DA3 bridges between them, DA3-metric depth, the same layout engine |
| Photo | Folder of per-room folders, 2–8 stills each | Any iPhone 15+ | Per-room metric depth and photo registration, then door-matching stitch |

## Install (about 10 minutes on a clean Mac/Linux machine)
```bash
git clone <repo> && cd <repo>
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install torch==2.8.0 torchvision==0.23.0 transformers==4.56.1 kornia==0.8.1 pycolmap==4.2.1
scripts/install_learned.sh            # Depth Anything 3 (no-deps install) + model weights into weights/
```
The LiDAR tier needs only the first `pip install`. Weights are fetched by script and never committed. They are: DA3METRIC-LARGE and DA3-BASE (Apache-2.0), CLIP ViT-B/32 (MIT), and LoFTR indoor (Apache-2.0).

## Run
```bash
scan2plan run path/to/stray_scan_or.zip                 # LiDAR (auto-detected)
scan2plan run path/to/walkthrough.MOV --tier video
scan2plan run path/to/property_photos/ --tier photo     # one sub-folder per room
scan2plan run <capture> --no-drift                      # ablation: poses used as-is
scan2plan validate runs/<name>/plan.json
```
Outputs go to `runs/<name>_<tier>/`: `plan.json`, `plan.png`, `plan.svg` and `log.txt`.

## Reproduce every reported number
```bash
scripts/fetch_data.sh SampleDataTotest                   # unpack raw captures into data/raw
python scripts/calibrate_metric_depth.py                 # DA3-metric vs LiDAR scale bias
python -m scan2plan.benchmark.run --tag baseline         # all gates, all tiers -> benchmark/results/baseline
python -m scan2plan.benchmark.run --tag after_fix        # fix-loop "after" run (see docs/fix_loop.md)
```
Learned-model outputs are cached under `data/cache/` in deterministic npz files. Delete that folder to force the live path.

## Results on the provided sample data
`scripts/run_sample_data.sh SampleDataTotest final` runs everything below in about 25 min on an M1 with warm caches. Committed outputs (JSON/PNG/SVG/log per run) are in `results/sample_data/`, and the report is `benchmark/results/final/report.md`.

| capture | tier | rooms | floor area (95% CI) | wall clock |
|---|---|---|---|---|
| single_room | LiDAR | 3 | 19.2 ± 1.7 m² | 39 s |
| single_scan_floor_only | LiDAR | 6 | 47.4 ± 2.9 m² | 124 s |
| single_scan_with_ceiling | LiDAR | 7 | 49.1 ± 3.3 m² | 154 s |
| (same 3) | video | 2 / 3 / 4 | 14.6 / 30.8 / 35.4 m² | 83–196 s (cached SfM+depth; cold about 15–20 min) |
| (same 3) | photo | 2 / 5 / 6 | 3.4 / 26.9 / 11.8 m² | 2–6 s (cached; cold about 1 min/room) |

- **Same flat, two LiDAR sessions:** stitched footprint IoU **0.848** with drift correction vs 0.791 without. The two areas, 47.4 and 49.1 m², are mutually consistent within their intervals.
- **Repeatability gate:** fails (23% of walls within 1 cm). The cause is that each walk divides the open-plan area into rooms differently (`docs/fix_loop.md`). The fully observed room agrees within 1.4 cm.
- **Calibration:** intervals cover the cross-session differences 86% of the time.
- **Video and photo** reconstruct a reduced part of the flat from these LiDAR-style sweeps (close range, fast turns). Their areas are under-estimates against the LiDAR reference, which is reported, not hidden.
- **No ground truth in the sample data.** For absolute accuracy we added an **external laser-scanner benchmark**: ARKitScenes, Apple's dataset of iPad Pro LiDAR walks with Faro laser scans of the same rooms (CC BY-NC-SA). We used 5 walks over 2 visits, with no parameters tuned on the held-out visit. Run it with `python scripts/arkitscenes_gt.py <visit> <videos...>`; the reports are `benchmark/arkitscenes/report_*.md`.
  - **Ceiling height:** errors of −0.6 to −3.7 cm on 5 rooms, median −2.0 cm. The result is *repeatable but biased* (about −0.9%, consistent with LiDAR depth scale). 2 of 5 rooms are within 1.5 cm; 4 of 5 are inside the 95% CI.
  - **Wall lengths** (small cluttered bathrooms, 14 walls with laser GT): median error 7–19 cm, 7% within 1 cm, and 79% of the 95% CIs contain the GT.

## Docs
- `docs/capture_protocol.md`: the one-page protocol a non-engineer follows. `docs/device_matrix.md`: which tier runs on which hardware, and the accuracy each tier delivers.
- `docs/compliance_matrix.md`: requirement → file → artifact → status.
- `docs/technical_report.md`: architecture, drift, error budget, calibration, fix loop, failure modes.
- `docs/fix_loop.md`: the fix declaration, with before/after numbers.
- `benchmark/results/*/report.md`: the generated benchmark reports.
