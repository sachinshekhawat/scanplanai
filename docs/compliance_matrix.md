# Compliance matrix

Status: ✅ done and evidenced · 🟡 partial (limitation stated) · ❌ not done (reason given)

| # | Requirement (case study) | File / path | Artifact | Status |
|---|---|---|---|---|
| 1 | Capture route: stock app + one-page protocol | `docs/capture_protocol.md` | protocol for Stray Scanner (LiDAR) and native Camera (video/photo) | ✅ |
| 2 | Three tiers, same output contract | `src/scan2plan/{lidar,video,photo}/`, `plan.py` | `runs/bench_*/<capture>_<tier>/plan.json` | 🟡 all three tiers produce a validated plan on all 3 sample captures (`results/sample_data/`); video/photo under-reconstruct extent vs LiDAR (reported) |
| 3 | Device matrix with honest accuracy | `docs/device_matrix.md` | table | ✅ |
| 4 | Dimensioned per-room plan: walls, ceiling height, floor area, openings | `lidar/rooms.py`, `lidar/to_plan.py` | `rooms[].walls/floor_area/ceiling_height`, `openings[]` | ✅ (ceiling marked `observed:false` when not scanned) |
| 5 | Stitched multi-room plan with adjacency | `plan.py::adjacency`, `photo/stitch.py` | `property.adjacency`, `property.overlaps`, `plan.png/svg` | ✅ LiDAR/video; 🟡 photo (door-matching stitch, shared-wall fallback flagged in notes) |
| 6 | Per-surface damage regions, class + metric extent | `damage/detect.py` | `damage[]` with surface id, class, confidence, area/width/height CIs | 🟡 zero-shot (CLIP), no damage-labelled benchmark room in the sample data |
| 7 | Concealed-damage flags with the rule that fired | `damage/__init__.py::RULES` | `concealed_damage_flags[].rule/rationale/evidence` | ✅ |
| 8 | Scope line items keyed to surfaces | `damage/__init__.py::_scope_for` | `scope[]` (surface id, action, quantity CI, because) | ✅ |
| 9 | Confidence interval on every measurement | `measure.py`, `schema.py` (validator checks every interval) | `sigma`, `ci95`, `observed`, `method` on each number | ✅ |
| 10 | One command per capture | `cli.py` | `scan2plan run <capture> [--tier] [--no-drift]` | ✅ |
| 11 | JSON to published schema | `schema/plan.schema.json`, `schema.py` | validated on every run | ✅ |
| 12 | Rendered plan | `render.py` | `plan.png`, `plan.svg` | ✅ |
| 13 | Drift accountability + on/off ablation | `lidar/drift.py`, `--no-drift` | `benchmark/results/*/metrics.json::drift_ablation` (cross-session IoU 0.836 on / 0.786 off) | ✅ |
| 14 | Benchmark composition (multi-room, damage room, all tiers, repeat room, GT) | `benchmark/run.py`, `benchmark/ground_truth_template.csv` | sample data covers multi-room + a repeat pair at LiDAR tier | 🟡 no tape GT and no staged-damage room in the provided data; the harness reads `benchmark/ground_truth.csv` when supplied |
| 15 | Gates: openings, ceiling, repeatability, drift, photo stitch | `benchmark/run.py::metrics` | `benchmark/results/{baseline,after_fix,final}/report.md` | 🟡 computed; most fail on sample data (numbers in report) |
| 16 | Calibration scored at every tier | `benchmark/run.py::coverage`, `scripts/calibrate_metric_depth.py` | coverage95 per tier; `benchmark/results/metric_depth_calibration.json` | ✅ |
| 17 | Head-to-head vs consumer app | n/a | n/a | ❌ needs our own capture of 2 rooms with Polycam/magicplan; the sample data carries no app export |
| 18 | Fix loop (declaration, before/after, diff) | `docs/fix_loop.md` | `benchmark/results/baseline` vs `after_fix` | 🟡 correct-but-incomplete root cause; shipped calibration fix (coverage 0.30 → 0.80); gate still failing, explained |
| 19 | Process evidence (commit history) | git log | incremental commits through the build | ✅ |
| 20 | README to running in < 15 min, reproduction bundle | `README.md`, `scripts/run_sample_data.sh`, `scripts/*` | install + fetch + run + benchmark commands; deterministic caches in `data/cache` | ✅ |
| 21 | Technical report ≤ 6 pages | `docs/technical_report.md` | report | ✅ |
| 22 | Raw benchmark data | `SampleDataTotest/*.zip` (raw Stray logs), `data/photo_sets/*/MANIFEST.txt` | provided captures + derivation manifests | 🟡 no GT / app exports (see 14, 17) |
| 23 | Mirrors, glass, wet surfaces, low light covered | `docs/capture_protocol.md`, `docs/technical_report.md §7` | protocol rules + failure-mode section; LiDAR confidence masking + flying-pixel rejection in `lidar/fusion.py` | 🟡 mitigations, not measured |
| 24 | Weights fetched by script, no own infrastructure | `scripts/install_learned.sh`, `.gitignore` | weights in `weights/` (ignored) | ✅ |
| 25 | Run on the provided sample data (email) | `scripts/run_sample_data.sh` | `results/sample_data/*/plan.{json,png,svg}`, `benchmark/results/final/` | ✅ |
| 26 | Tests | `tests/` | 9 tests: schema, intervals, geometry primitives, loader conventions | ✅ |
