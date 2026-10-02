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

## Docs
- `docs/capture_protocol.md`: the one-page protocol a non-engineer follows. `docs/device_matrix.md`: which tier runs on which hardware, and the accuracy each tier delivers.
- `docs/compliance_matrix.md`: requirement → file → artifact → status.
- `docs/technical_report.md`: architecture, drift, error budget, calibration, fix loop, failure modes.
- `docs/fix_loop.md`: the fix declaration, with before/after numbers.
- `benchmark/results/*/report.md`: the generated benchmark reports.
