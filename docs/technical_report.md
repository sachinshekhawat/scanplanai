# scan2plan: technical report

All numbers below come from `benchmark/results/{baseline,after_fix}` and `benchmark/results/metric_depth_calibration.json`. The provided sample data has **no tape or laser ground truth**. LiDAR accuracy is therefore shown through cross-session repeatability, and the video and photo tiers are scored against the LiDAR result of the same capture ("LiDAR-referenced", which is not ground truth).

## 1. Architecture
```
capture ─► tier front-end ─► posed metric RGB-D frames (PseudoCapture / StrayCapture interface)
            LiDAR : Stray depth + ARKit poses, drift-corrected (submap pose graph)
            video : COLMAP SfM fragments ─ DA3 bridges ─ DA3-metric depth, gravity, scale
            photo : per-room DA3-metric depth + SIFT/LoFTR registration ─ door-matching stitch
        ─► shared layout engine: fusion (normals, flying-pixel rejection, voxel hash)
             → floor/ceiling planes → Manhattan yaw → plan-view ray carving (free space)
             → structural walls (height coverage) → wall-face runs (sub-cm offsets)
             → axis-closure room sealing → openings (jamb-snapped) → rectilinear polygons
        ─► Plan (rooms, edges with σ, openings) ─► damage / rules / scope ─► JSON (schema v1) + PNG/SVG
```
- **One code path for geometry:** every tier feeds the same layout engine, so improvements and bugs show up in all tiers.
- **Isolated learned models:** torch, pycolmap and open3d ship clashing libomp builds on macOS. Each learned model therefore runs in a child process (`learned/da3_worker.py`, `video/sfm_worker.py`, `learned/loftr_worker.py`, `damage/clip_worker.py`) and returns plain npz files, which are also the deterministic replay cache under `data/cache/`.

## 2. Tier design and device matrix
See `docs/device_matrix.md`. Key facts measured on the sample data:
- **Stray conventions:**
  - OpenCV camera frame. The floor peak holds 17% of points in a 4 cm slab; the ARKit −z convention gives 4%.
  - Frames are stored rotated 90°, so monocular depth must run on upright frames. Before the fix the scale error was −63%; after it, per-fragment scale is within 2–15%.
  - `rgb.mp4` lags the odometry by a time-varying **0.1–1.5 s** (`io/sync.py`, e.g. +27…+69 frames on `with_ceiling`). The LiDAR tier never uses RGB for geometry. The damage stage and every RGB-vs-pose evaluation apply the measured offset.
- **Video:**
  - SIFT with a low peak threshold (≈3,400 keypoints/frame instead of 440) plus sequential matching. Interiors break SfM into fragments (6 on `single_room`).
  - Fragments are joined by **DA3-BASE multi-view poses on the gap window**. Measured relative-rotation error on upright gap windows is 0.6–1.3° median, with one gap failing at 88°. Wall-raster correlation is the fallback.
  - Metric scale comes from **DA3METRIC-LARGE**.
  - Trajectory vs ARKit on `single_room`: 0.57 m RMSE, scale error +2%.
- **Photo:**
  - Per-room metric depth. Registration uses SIFT, with LoFTR for pairs SIFT misses, and picks a model per pair: pure rotation (turn in place, per the protocol), rigid 3D–3D, or essential matrix only with ≥ 40 inliers. A max-spanning tree, then rooms are stitched by door matching in folder order.
  - Evaluation photo sets are frames of the Stray video (`scripts/make_photo_set.py`, disclosed in each `MANIFEST.txt`). They are motion-blurred, close-range and low-overlap, which is harder than protocol photos. Only 2–4 of 4–8 photos register per room.

## 3. Drift handling
`lidar/drift.py` builds a **submap pose graph**: one 2.5 m-of-walk submap per node, with a 2D rigid correction and a vertical offset per submap. Gauss-Newton solves for:
- **Loop closure:** point-to-plane wall-to-wall matches between non-adjacent submaps with the same normal, coarse-to-fine gates 0.30 → 0.05 m, Huber weighting.
- **Plane-anchored Manhattan yaw:** each submap's wall normals are pulled onto the global dominant axes.
- **Odometry prior:** about 1° and 2 cm per submap.
- **Floor anchor:** each submap's floor equals the global floor.

Corrections are interpolated per frame by path length and applied before full fusion. `--no-drift` is the ablation (poses used as-is).

| capture | submaps | path | max yaw corr. | max shift | loop residual (first → last iteration) |
|---|---|---|---|---|---|
| with_ceiling | 40 | 98.8 m | 1.83° | 10.3 cm | 12.8 → 9.4 mm |
| floor_only | 22 | 53.8 m | 3.31° | 10.3 cm | 30.5 → 14.8 mm |

Ablation: the same flat captured in two sessions has a stitched-footprint IoU of **0.836 with drift correction vs 0.786 without**, and cross-session wall overlap rises from 0.21 to 0.61 on raw wall pixels. The on/off footprints are in `metrics.json::drift_ablation`.

## 4. Error budget (1-σ, `measure.py`)
| term | LiDAR | video | photo |
|---|---|---|---|
| wall face position | MAD of face inliers / √(n/20) ⊕ 4 mm system | ⊕ 2 cm face floor | ⊕ 2 cm face floor |
| face ambiguity (fix loop) | ½ spread of competing compatible faces within 0.6 m | same | same |
| edge with no wall face | 4 cm | 4 cm | 4 cm |
| length scale | 0.2 % | **4 %** (calibrated, §5) | **6.5 %** |
| opening jamb | 6 mm snapped / 3 cm free-space chord | ⊕ face floor ⊕ scale | wall-gap 4 cm ⊕ scale |
| room placement in stitch | n/a | n/a | 5 cm per room |
| ceiling | plane spread / √n ⊕ 3 mm ⊕ scale; unobserved → prior 2.60 ± 0.30 m (`observed:false`) | ×2 ⊕ scale | ×2 ⊕ scale |

Lengths combine the two perpendicular bounding faces and the scale term in quadrature. Areas are first-order propagated, plus a 2× scale term.

## 5. Calibration analysis
- **Metric depth (video/photo):**
  - DA3METRIC / LiDAR depth ratio on upright frames: 0.934, 0.940 and 0.854 per capture; pooled **0.933**.
  - Per-frame log-σ 0.118; capture-to-capture spread 0.039 → video scale σ 4%.
  - The photo tier averages only 4–6 frames per room → 6.5%.
- **Interval coverage (fraction of observed errors inside the stated 95% CI):**

| tier / check | n | coverage95 | median \|z\| | verdict |
|---|---|---|---|---|
| LiDAR cross-session walls, baseline | 20 | 0.30 | 5.2 | over-confident |
| LiDAR cross-session walls, after fix | 20 | **0.80** | **0.75** | close to calibrated (target 0.95) |
| video vs LiDAR, single_room walls | 9 | 0.44 | — | over-confident (geometry errors > scale model) |
| video vs LiDAR, single_room floor area | 1 | in CI (+10.2%) | — | ok |
| video / photo, with_ceiling | 1–2 | 0.0 | — | wrong geometry (footprint IoU 0.39 / 0.19) |

Confident garbage on thin input is the risk the brief names. The video and photo tiers are not yet calibrated on multi-room walks. Their intervals cover scale error but not gross registration failures. §7 lists this as the top open item.

## 6. Fix loop (`docs/fix_loop.md`)
- **Worst LiDAR gate:** repeatability. 25% of walls pass, median 13.8 cm.
- **Hypothesis:** walls bounded only by free space. This was tested and is **incomplete**: 16 of 20 walls were face-bounded and still disagreed. The dominant cause is that each session partitions the open plan into rooms differently.
- **Declared fix:** outward face snapping. Shipped behind a flag; no gain. Six variants are tabulated; my prediction was badly wrong.
- **Shipped fix:** a per-edge face-ambiguity term. The geometry is unchanged, so the gate still fails on agreement. Calibration moves from coverage 0.30 to 0.80. The plan is now *unrepeatable but honestly uncertain*, no longer over-confident.
- **Next fix:** a shared layout prior across sessions. Evidence: the fully observed room already agrees within 1.4 cm.

## 7. Known failure modes
1. **Room partition is session-dependent** in open-plan areas (fix-loop finding). The intervals now reflect it.
2. **Video SfM fragments** on close-range, fast-turn walks. `single_scan_floor_only` video fails ("no usable SfM fragment"), and `with_ceiling` registers about 1,300 frames but stitches with footprint IoU 0.39. Protocol mitigation: 1.5 m from walls, turns ≤ 10 s per 360°.
3. **Photo registration** with < 1/3 overlap: 2–4 of 4–8 photos connect. The protocol requires 1/3 overlap and turning in place. The video-frame photo sets used here violate both, so photo numbers are a lower bound.
4. **Ceiling**: only `with_ceiling` observes it. Per-room values range from 2.28 to 3.08 m, so some rooms may lock onto bulkheads or high cabinet tops. With no ground truth this can't be resolved; those values are reported with σ ≈ 6–7 mm, which may be over-confident.
5. **Damage**:
   - Zero-shot CLIP, with no labelled damage room in the data.
   - Straight grout lines and tile edges are rejected by a tortuosity test; the real bathroom crack is detected only when its view is assigned to a plan surface.
   - Recall and precision are unmeasured.
6. **Mirrors / glass / wet floors**: LiDAR returns through glass and from mirrors appear as phantom free space beyond a wall. Mitigations:
   - confidence-2 depth only;
   - flying-pixel rejection;
   - free space requires ≥ 2 views;
   - the protocol asks for sticky notes on mirrors and glass.

   Low light lowers RGB sharpness, so blurred views are skipped by damage detection.
7. **Not delivered**: the head-to-head against a consumer app, and tape ground truth. Both need our own captures (the protocol and GT template are ready).
