# Benchmark report (auto-generated)

No tape/laser ground truth in the provided sample data. LiDAR is the reference for video/photo (LiDAR-referenced error, not GT). LiDAR absolute accuracy gates need benchmark/ground_truth.csv.

## Gates

| gate | criterion | value | pass |
|---|---|---|---|
| repeatability | |diff| <= max(1 cm, 0.5%) per wall | 0.227 | False |
| drift_accountability | drift handling stated + on/off ablation | {"on": 0.8478581052291211, "off": 0.7911485870319735} | True |
| ceiling_height | <= 1.5 cm vs GT; spread <= 1 cm | "no GT" | None |
| video_tier | walls within +-3% (LiDAR-referenced) | 0.067 | False |
| photo_tier | walls within +-8%, footprint +-8%, no overlaps | 0.056 | False |
| opening_widths | <= 2 cm on >= 85% (miss & phantom count as fail); cross-session proxy | 0.125 | False |

## Repeatability (two sessions, LiDAR tier)

Footprint IoU after alignment 0.848; rooms [7, 6]; matched room pairs 6; wall pass rate 0.22727272727272727; median |diff| 0.10309999999999997; calibration {'n': 22, 'coverage95': 0.9090909090909091, 'median_abs_z': 0.7715352930663049}

| room A | wall | len A (m) | len B (m) | diff (cm) | pass |
|---|---|---|---|---|---|
| R1 | R1-W1 | 3.091 | 3.090 | -0.1 | True |
| R1 | R1-W2 | 2.864 | 2.877 | +1.4 | True |
| R1 | R1-W3 | 3.091 | 3.090 | -0.1 | True |
| R1 | R1-W4 | 2.864 | 2.877 | +1.4 | True |
| R2 | R2-W1 | 2.211 | 1.804 | -40.7 | False |
| R2 | R2-W2 | 3.469 | 3.010 | -45.9 | False |
| R2 | R2-W4 | 2.782 | 2.306 | -47.6 | False |
| R2 | R2-W5 | 0.566 | 0.853 | +28.6 | False |
| R2 | R2-W6 | 0.687 | 0.704 | +1.7 | False |
| R3 | R3-W1 | 2.765 | 2.750 | -1.6 | False |
| R3 | R3-W2 | 3.100 | 3.131 | +3.1 | False |
| R3 | R3-W7 | 2.028 | 1.972 | -5.5 | False |
| R3 | R3-W8 | 4.061 | 3.904 | -15.7 | False |
| R4 | R4-W2 | 0.880 | 1.704 | +82.4 | False |
| R4 | R4-W3 | 1.369 | 1.520 | +15.1 | False |
| R4 | R4-W4 | 0.880 | 1.704 | +82.4 | False |
| R5 | R5-W1 | 1.913 | 2.299 | +38.5 | False |
| R5 | R5-W2 | 2.275 | 2.287 | +1.2 | False |
| R5 | R5-W3 | 1.913 | 2.299 | +38.5 | False |
| R7 | R7-W3 | 2.714 | 0.805 | -190.9 | False |
| R7 | R7-W5 | 3.389 | 3.372 | -1.7 | True |
| R7 | R7-W6 | 3.322 | 3.303 | -2.0 | False |

## Drift ablation

```
{
 "single_scan_floor_only": {
  "on": {
   "n_rooms": 6,
   "area_m2": 48.0838,
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
   "area_m2": 46.9609,
   "overlaps": 0
  }
 },
 "single_scan_with_ceiling": {
  "on": {
   "n_rooms": 7,
   "area_m2": 49.7544,
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
   "area_m2": 49.8745,
   "overlaps": 0
  }
 },
 "cross_session_footprint_iou": {
  "on": 0.8478581052291211,
  "off": 0.7911485870319735
 }
}
```

## Cross-tier (LiDAR-referenced)

| capture/tier | footprint IoU | rooms ref/tier | area rel err | area in CI | walls | wall pass | median rel | calib cov95 |
|---|---|---|---|---|---|---|---|---|
| single_room_video | 0.43 | [3, 2] | -10.3% | True | 3 | 0.0 | 0.24717935954869757 | 0.3333333333333333 |
| single_room_photo | 0.17 | [3, 2] | -83.3% | False | 1 | 0.0 | 0.6411651578404239 | 0.0 |
| single_scan_floor_only_video | 0.34 | [6, 4] | -12.1% | True | 5 | 0.0 | 0.810314659390848 | 0.0 |
| single_scan_floor_only_photo | 0.40 | [6, 5] | -38.2% | False | 6 | 0.16666666666666666 | 0.4862700975233485 | 0.5 |
| single_scan_with_ceiling_video | 0.37 | [7, 3] | +14.5% | True | 5 | 0.2 | 0.3242514647208943 | 0.4 |
| single_scan_with_ceiling_photo | 0.25 | [7, 6] | -69.9% | False | 4 | 0.0 | 0.6694988892936153 | 0.75 |

## Openings (cross-session)

```
{"matched": 4, "missed": 1, "phantom": 3, "within_2cm": 1, "diffs_m": [-0.11220000000000008, 0.0129999999999999, 0.4939, -0.37520000000000003]}
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
  "n_regions": 2,
  "classes": [
   "crack",
   "hole"
  ],
  "flags": []
 },
 "single_room_video": {
  "n_regions": 3,
  "classes": [
   "crack",
   "mold"
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
  "n_regions": 2,
  "classes": [
   "crack",
   "mold"
  ],
  "flags": []
 },
 "single_scan_floor_only_lidar_nodrift": {
  "n_regions": 4,
  "classes": [
   "crack",
   "mold"
  ],
  "flags": [
   "R5_long_crack"
  ]
 },
 "single_scan_floor_only_video": {
  "n_regions": 1,
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
  "n_regions": 6,
  "classes": [
   "crack",
   "mold"
  ],
  "flags": [
   "R1_ceiling_water",
   "R5_long_crack"
  ]
 },
 "single_scan_with_ceiling_lidar_nodrift": {
  "n_regions": 12,
  "classes": [
   "crack",
   "hole",
   "mold",
   "water_stain"
  ],
  "flags": [
   "R1_ceiling_water",
   "R1_ceiling_water",
   "R6_crack_at_opening",
   "R6_crack_at_opening",
   "R5_long_crack",
   "R1_ceiling_water"
  ]
 },
 "single_scan_with_ceiling_video": {
  "n_regions": 1,
  "classes": [
   "mold"
  ],
  "flags": [
   "R1_ceiling_water"
  ]
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
| single_room_lidar | 40 |
| single_room_lidar_nodrift | 37 |
| single_room_video | 89 |
| single_room_photo | 1 |
| single_scan_floor_only_lidar | 115 |
| single_scan_floor_only_lidar_nodrift | 110 |
| single_scan_floor_only_video | 164 |
| single_scan_floor_only_photo | 3 |
| single_scan_with_ceiling_lidar | 158 |
| single_scan_with_ceiling_lidar_nodrift | 139 |
| single_scan_with_ceiling_video | 136 |
| single_scan_with_ceiling_photo | 4 |
