# Benchmark report (auto-generated)

No tape/laser ground truth in the provided sample data. LiDAR is the reference for video/photo (LiDAR-referenced error, not GT). LiDAR absolute accuracy gates need benchmark/ground_truth.csv.

## Gates

| gate | criterion | value | pass |
|---|---|---|---|
| repeatability | |diff| <= max(1 cm, 0.5%) per wall | 0.227 | False |
| drift_accountability | drift handling stated + on/off ablation | {"on": 0.847859009050916, "off": 0.7911490918534975} | True |
| ceiling_height | <= 1.5 cm vs GT; spread <= 1 cm | "no GT" | None |
| video_tier | walls within +-3% (LiDAR-referenced) | 0.056 | False |
| photo_tier | walls within +-8%, footprint +-8%, no overlaps | 0.167 | False |
| opening_widths | <= 2 cm on >= 85% (miss & phantom count as fail); cross-session proxy | 0.125 | False |

## Repeatability (two sessions, LiDAR tier)

Footprint IoU after alignment 0.848; rooms [7, 6]; matched room pairs 6; wall pass rate 0.22727272727272727; median |diff| 0.10239999999999994; calibration {'n': 22, 'coverage95': 0.8636363636363636, 'median_abs_z': 0.8239765786424784}

| room A | wall | len A (m) | len B (m) | diff (cm) | pass |
|---|---|---|---|---|---|
| R1 | R1-W1 | 3.070 | 3.070 | -0.1 | True |
| R1 | R1-W2 | 2.844 | 2.858 | +1.4 | True |
| R1 | R1-W3 | 3.070 | 3.070 | -0.1 | True |
| R1 | R1-W4 | 2.844 | 2.858 | +1.4 | True |
| R2 | R2-W1 | 2.197 | 1.792 | -40.4 | False |
| R2 | R2-W2 | 3.446 | 2.990 | -45.6 | False |
| R2 | R2-W4 | 2.764 | 2.291 | -47.3 | False |
| R2 | R2-W5 | 0.563 | 0.847 | +28.4 | False |
| R2 | R2-W6 | 0.682 | 0.699 | +1.7 | False |
| R3 | R3-W1 | 2.747 | 2.731 | -1.6 | False |
| R3 | R3-W2 | 3.079 | 3.110 | +3.1 | False |
| R3 | R3-W7 | 2.014 | 1.959 | -5.5 | False |
| R3 | R3-W8 | 4.034 | 3.878 | -15.6 | False |
| R4 | R4-W2 | 0.874 | 1.693 | +81.9 | False |
| R4 | R4-W3 | 1.360 | 1.510 | +15.0 | False |
| R4 | R4-W4 | 0.874 | 1.693 | +81.9 | False |
| R5 | R5-W1 | 1.901 | 2.283 | +38.3 | False |
| R5 | R5-W2 | 2.260 | 2.272 | +1.2 | False |
| R5 | R5-W3 | 1.901 | 2.283 | +38.3 | False |
| R7 | R7-W3 | 2.696 | 0.800 | -189.6 | False |
| R7 | R7-W5 | 3.366 | 3.349 | -1.7 | True |
| R7 | R7-W6 | 3.300 | 3.281 | -1.9 | False |

## Drift ablation

```
{
 "single_scan_floor_only": {
  "on": {
   "n_rooms": 6,
   "area_m2": 47.4416,
   "overlaps": 0,
   "drift": {
    "method": "submap pose graph: wall loop-closure + Manhattan yaw anchor + odometry prior + floor anchor",
    "enabled": true,
    "n_submaps": 22,
    "n_keyframes": 567,
    "path_m": 53.76,
    "max_abs_theta_deg": 3.3095,
    "max_abs_shift_m": 0.1028,
    "max_abs_dy_m": 0.025,
    "loop_residual_m_first_last": [
     0.0305,
     0.0148
    ],
    "n_loop_matches_found": true
   }
  },
  "off": {
   "n_rooms": 6,
   "area_m2": 46.3337,
   "overlaps": 0
  }
 },
 "single_scan_with_ceiling": {
  "on": {
   "n_rooms": 7,
   "area_m2": 49.0899,
   "overlaps": 0,
   "drift": {
    "method": "submap pose graph: wall loop-closure + Manhattan yaw anchor + odometry prior + floor anchor",
    "enabled": true,
    "n_submaps": 40,
    "n_keyframes": 1090,
    "path_m": 98.8448,
    "max_abs_theta_deg": 1.8341,
    "max_abs_shift_m": 0.1029,
    "max_abs_dy_m": 0.0043,
    "loop_residual_m_first_last": [
     0.0128,
     0.0094
    ],
    "n_loop_matches_found": true
   }
  },
  "off": {
   "n_rooms": 6,
   "area_m2": 49.2085,
   "overlaps": 0
  }
 },
 "cross_session_footprint_iou": {
  "on": 0.847859009050916,
  "off": 0.7911490918534975
 }
}
```

## Cross-tier (LiDAR-referenced)

| capture/tier | footprint IoU | rooms ref/tier | area rel err | area in CI | walls | wall pass | median rel | calib cov95 |
|---|---|---|---|---|---|---|---|---|
| single_room_video | 0.39 | [3, 2] | -23.9% | False | 6 | 0.16666666666666666 | 0.31866379572253994 | 0.6666666666666666 |
| single_room_photo | 0.18 | [3, 2] | -82.1% | False | 2 | 0.5 | 0.23903777740325943 | 1.0 |
| single_scan_floor_only_video | 0.33 | [6, 3] | -35.1% | False | 5 | 0.0 | 0.17994739285161687 | 0.6 |
| single_scan_floor_only_photo | 0.38 | [6, 5] | -43.3% | False | 2 | 0.0 | 1.4261891523092993 | 0.5 |
| single_scan_with_ceiling_video | 0.43 | [7, 4] | -28.0% | False | 5 | 0.0 | 0.27682976377038726 | 0.4 |
| single_scan_with_ceiling_photo | 0.23 | [7, 6] | -76.1% | False | 1 | 0.0 | 0.2031880704460727 | 1.0 |

## Openings (cross-session)

```
{"matched": 4, "missed": 1, "phantom": 3, "within_2cm": 1, "diffs_m": [-0.11139999999999994, 0.012899999999999912, 0.4905999999999999, -0.3727]}
```

## Ceiling

Only single_scan_with_ceiling observes the ceiling; the other session did not scan it, so cross-capture spread is not measurable on sample data (reported as unobserved prior).

## Damage

```
{
 "single_room_lidar": {
  "n_regions": 1,
  "classes": [
   "crack"
  ],
  "flags": []
 },
 "single_room_lidar_nodrift": {
  "n_regions": 1,
  "classes": [
   "crack"
  ],
  "flags": []
 },
 "single_room_video": {
  "n_regions": 3,
  "classes": [
   "crack"
  ],
  "flags": [
   "R5_long_crack"
  ]
 },
 "single_room_photo": {
  "n_regions": 0,
  "classes": [],
  "flags": []
 },
 "single_scan_floor_only_lidar": {
  "n_regions": 3,
  "classes": [
   "crack",
   "hole",
   "mold"
  ],
  "flags": []
 },
 "single_scan_floor_only_lidar_nodrift": {
  "n_regions": 2,
  "classes": [
   "crack",
   "mold"
  ],
  "flags": []
 },
 "single_scan_floor_only_video": {
  "n_regions": 2,
  "classes": [
   "crack"
  ],
  "flags": []
 },
 "single_scan_floor_only_photo": {
  "n_regions": 0,
  "classes": [],
  "flags": []
 },
 "single_scan_with_ceiling_lidar": {
  "n_regions": 10,
  "classes": [
   "crack",
   "hole",
   "mold",
   "water_stain"
  ],
  "flags": [
   "R1_ceiling_water",
   "R1_ceiling_water",
   "R5_long_crack"
  ]
 },
 "single_scan_with_ceiling_lidar_nodrift": {
  "n_regions": 7,
  "classes": [
   "crack",
   "mold",
   "water_stain"
  ],
  "flags": [
   "R1_ceiling_water",
   "R1_ceiling_water",
   "R6_crack_at_opening",
   "R5_long_crack",
   "R2_low_wall_moisture"
  ]
 },
 "single_scan_with_ceiling_video": {
  "n_regions": 1,
  "classes": [
   "mold"
  ],
  "flags": []
 },
 "single_scan_with_ceiling_photo": {
  "n_regions": 0,
  "classes": [],
  "flags": []
 }
}
```

## Timing (s, wall clock per capture)

| run | s |
|---|---|
| single_room_lidar | 39 |
| single_room_lidar_nodrift | 37 |
| single_room_video | 83 |
| single_room_photo | 2 |
| single_scan_floor_only_lidar | 124 |
| single_scan_floor_only_lidar_nodrift | 112 |
| single_scan_floor_only_video | 196 |
| single_scan_floor_only_photo | 4 |
| single_scan_with_ceiling_lidar | 154 |
| single_scan_with_ceiling_lidar_nodrift | 137 |
| single_scan_with_ceiling_video | 181 |
| single_scan_with_ceiling_photo | 6 |
