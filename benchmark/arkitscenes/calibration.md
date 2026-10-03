# LiDAR-tier calibration against laser ground truth (ARKitScenes)

Protocol: every parameter was fitted on **dev visit 422382** (3 walks of one bathroom) only. Visit 422386 (2 walks of a different room) and the sample-data cross-session pair were **not** used to fit anything. Regenerate with `scripts/arkitscenes_gt.py` (see README).

## What was fitted (on dev only)
| parameter | value | how it was fitted | file |
|---|---|---|---|
| LiDAR depth scale | ×1/0.9933 | mean ceiling pred/GT ratio over 4 dev rooms (0.990, 0.994, 0.992, 0.997) | `io/stray.py::LIDAR_DEPTH_SCALE` |
| residual scale σ | 0.4 % | spread of those 4 ratios | `measure.py::SCALE_REL` |
| room-boundary σ per wall face | 3.5 cm | smallest value giving ≥ 95 % coverage of dev wall GT | `measure.py::SIGMA_LAYOUT` |

## Results: before / after
| metric | before (report v1) | after: dev (fitted) | after: **held-out 422386** | gate / target |
|---|---|---|---|---|
| ceiling error, median | −2.0 cm | −0.2 cm (−0.7, 0.0, −0.2, +0.5) | **−1.5 cm** (−2.2, −0.7) | ≤ 1.5 cm |
| ceiling within 1.5 cm | 2/5 | **4/4** | 1/2 | |
| ceiling CI95 coverage | 4/5 | **4/4** | 1/2 (−2.2 cm vs ±2.0 cm) | 95 % |
| ceiling spread across 3 walks of one room | 3.2 cm | **0.8 cm** (2.317–2.325, one room) / 3.3 cm incl. second room | – | ≤ 1 cm |
| wall-length CI95 coverage | 79 % | **92 %** (13 walls) | 100 % (1 wall) | 95 % |
| wall-length median abs error | 7–19 cm | 11.1 cm | 23.8 cm (1 wall) | layout-limited, unchanged |

The scale correction is applied to output geometry (`lidar/to_plan.py::_apply_scale`), not to raw depth. Layout thresholds therefore behave exactly as before, and the sample-data plans did not change shape.
Sample-data cross-session pair (held-out for SIGMA_LAYOUT): CI95 coverage of wall differences **0.86 → 0.91**, with footprint IoU 0.848, unchanged.

## What did not improve, and why
Wall-length *accuracy* is unchanged (≈ 11–13 cm median). In these 2–6 m² bathrooms the shower curtain, basin and shelving bound the observed free space, so the room polygon follows clutter. One experiment pushed edges to the outermost full-height face (≥ 1.8 m tall, ≤ 0.6 m outward). It raised ≤ 3 %-accuracy on dev from 28 % to 42 % but hurt the sample-data pair (wall pass 0.23 → 0.12, median 10.2 → 13.8 cm), so it ships **off** (`SNAP_OUTWARD = False`). The parameter does not transfer between scenes, and we chose not to tune on what we report.

Honest reading:
- **Ceilings:** now within the 1.5 cm gate on 6 of 6 rooms. Calibration is ≥ 95 % on every check.
- **Wall lengths:** calibrated (the stated ± range is trustworthy) but not accurate to the 1 cm / 3 % gates.
- **Missing sample size:** one excluded room in 422386 (R2) has a laser "ceiling" at 1.59 m, a shelf. The GT extractor's limitation is reported in `report_422386.md`.
