# Fix loop declaration

## 1. Worst-performing gate (baseline run, `benchmark/results/baseline/report.md`)
**Repeatability (LiDAR tier).** Two sessions of the same apartment (`single_scan_floor_only` vs `single_scan_with_ceiling`):
- Only **25% of matched walls** agree within the gate of 1 cm / 0.5% (5 of 20).
- Median |difference| is **13.8 cm**.
- Interval coverage is 30% (median |z| 5.2), so the intervals are **unrepeatable and over-confident**.

The video and photo tiers score 0% on their gates. That is not the worst *LiDAR* number, and those tiers are judged against LiDAR, so their numbers are only as good as LiDAR repeatability. LiDAR has to be fixed first.

## 2. Root-cause hypothesis and evidence
Hypothesis: the error comes from **room-polygon extraction, not from sensing or drift**.
- The polygon is the contour of observed free space, simplified to axis-aligned edges.
- An edge only takes its position from a wall face when that face lies within 15 cm of it (`snap=0.15`).
- Where furniture, a doorway shadow or an unscanned corner stops the free space short of the wall, the edge stays at the free-space boundary.
- Short jogs over 8 cm survive as fake notches.
- Each session's walk observes different free space, so the same wall ends up at different places.

Evidence (from `runs/bench_baseline`):
- Walls where both sessions are bounded by a wall face on both ends ("face-to-face") differ by a median of **3.6 cm**. Walls with a free-space end differ by **107 cm** (both ends free) or **33 cm** (one end free).
- Session B has only 72% of walls bounded by a face (A: 87%). It also has far more polygon vertices for the same rooms: 8–10 vs 4–6 for R2, R3, R5 and R7.
- The simplest, fully observed room (R1↔R5) agrees within **0.1–1.4 cm** on all 4 walls, so sensing and drift are fine where geometry is observed.
- Drift is not the cause. Cross-session footprint IoU is 0.84 with drift correction vs 0.79 without, and the face-to-face residual (3.6 cm) sits within drift-corrected wall noise.

## 3. Fix and prediction
Fix (`src/scan2plan/lidar/rooms.py::room_polygon`):
- **(a) Snap outward:** an edge not bounded by a face is pushed outward to the nearest parallel wall face with the right orientation within 0.6 m. Free space never extends past a wall, so the true face can only be further out.
- **(b) Collapse jogs:** remove jogs < 0.35 m when both neighbouring edges snap to the same face.
- **(c) Regularise:** re-intersect the corners from the snapped faces.

Predicted after the fix, on the same pair:
- Walls bounded face-to-face on both sessions rise from ~70% to ≥ 90%.
- Repeatability median |diff| falls from 13.8 cm to ≤ 3 cm.
- Wall pass rate rises from 25% to about 45–60%.
- The full 1 cm gate stays out of reach: face-to-face walls still differ by 3.6 cm. Part of that is real (one session scans a wardrobe face, the other the wall behind it). I predict **meaningful movement short of the gate** and will report why.

The after-run is in `benchmark/results/after_fix/`; the before/after table is below.
