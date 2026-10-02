# Benchmark report (auto-generated)

No tape/laser ground truth in the provided sample data. LiDAR is the reference for video/photo (LiDAR-referenced error, not GT). LiDAR absolute accuracy gates need benchmark/ground_truth.csv.

## Gates

| gate | criterion | value | pass |
|---|---|---|---|
| repeatability | |diff| <= max(1 cm, 0.5%) per wall | 0.250 | False |
| drift_accountability | drift handling stated + on/off ablation | {"on": 0.8357076103646066, "off": 0.7860868533185664} | True |
| ceiling_height | <= 1.5 cm vs GT; spread <= 1 cm | "no GT" | None |
| video_tier | walls within +-3% (LiDAR-referenced) | 0.000 | False |
| photo_tier | walls within +-8%, footprint +-8%, no overlaps | 0.000 | False |
| opening_widths | <= 2 cm on >= 85% (miss & phantom count as fail); cross-session proxy | 0.125 | False |

## Repeatability (two sessions, LiDAR tier)

Footprint IoU after alignment 0.836; rooms [7, 6]; matched room pairs 6; wall pass rate 0.25; median |diff| 0.13794999999999985; calibration {'n': 20, 'coverage95': 0.3, 'median_abs_z': 5.222405728079716}

| room A | wall | len A (m) | len B (m) | diff (cm) | pass |
|---|---|---|---|---|---|
| R1 | R1-W1 | 3.070 | 3.070 | -0.1 | True |
| R1 | R1-W2 | 2.844 | 2.858 | +1.4 | True |
| R1 | R1-W3 | 3.070 | 3.070 | -0.1 | True |
| R1 | R1-W4 | 2.844 | 2.858 | +1.4 | True |
| R2 | R2-W1 | 2.197 | 1.792 | -40.4 | False |
| R2 | R2-W2 | 3.446 | 2.990 | -45.6 | False |
| R2 | R2-W5 | 0.563 | 0.683 | +12.0 | False |
| R2 | R2-W6 | 0.682 | 0.699 | +1.7 | False |
| R3 | R3-W1 | 2.747 | 2.636 | -11.1 | False |
| R3 | R3-W2 | 3.079 | 2.878 | -20.1 | False |
| R3 | R3-W7 | 2.014 | 1.959 | -5.5 | False |
| R3 | R3-W8 | 4.034 | 3.878 | -15.6 | False |
| R4 | R4-W2 | 1.129 | 1.693 | +56.4 | False |
| R4 | R4-W6 | 0.704 | 1.641 | +93.7 | False |
| R5 | R5-W1 | 1.901 | 1.651 | -25.0 | False |
| R5 | R5-W2 | 2.260 | 2.272 | +1.2 | False |
| R7 | R7-W1 | 0.670 | 1.084 | +41.4 | False |
| R7 | R7-W3 | 2.696 | 0.800 | -189.6 | False |
| R7 | R7-W5 | 3.366 | 3.349 | -1.7 | True |
| R7 | R7-W6 | 3.300 | 3.051 | -24.9 | False |

## Drift ablation

```
{
 "single_scan_floor_only": {
  "on": {
   "n_rooms": 6,
   "area_m2": 47.1093,
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
   "area_m2": 46.1714,
   "overlaps": 0
  }
 },
 "single_scan_with_ceiling": {
  "on": {
   "n_rooms": 7,
   "area_m2": 49.4362,
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
   "area_m2": 49.2319,
   "overlaps": 0
  }
 },
 "cross_session_footprint_iou": {
  "on": 0.8357076103646066,
  "off": 0.7860868533185664
 }
}
```

## Cross-tier (LiDAR-referenced)

| capture/tier | footprint IoU | rooms ref/tier | area rel err | area in CI | walls | wall pass | median rel | calib cov95 |
|---|---|---|---|---|---|---|---|---|
| single_room_video | 0.50 | [3, 2] | +10.2% | True | 9 | 0.0 | 0.3723707471948623 | 0.4444444444444444 |
| single_room_photo | FAILED: RuntimeError('photo tier: no room could be reconstructed') | | | | | | | |
| single_scan_floor_only_video | FAILED: RuntimeError('video tier: no usable SfM fragment (too little | | | | | | | |
| single_scan_floor_only_photo | FAILED: ValueError('The truth value of an array with more than one e | | | | | | | |
| single_scan_with_ceiling_video | 0.39 | [7, 3] | +97.7% | False | 1 | 0.0 | 0.4863311857913982 | 0.0 |
| single_scan_with_ceiling_photo | 0.19 | [7, 4] | -80.4% | False | 2 | 0.0 | 0.4650805308067196 | 0.0 |

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
   "crack",
   "water_stain"
  ],
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
  "n_regions": 2,
  "classes": [
   "crack",
   "mold"
  ],
  "flags": []
 },
 "single_scan_with_ceiling_lidar": {
  "n_regions": 9,
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
| single_room_lidar | 43 |
| single_room_lidar_nodrift | 38 |
| single_room_video | 44 |
| single_scan_floor_only_lidar | 123 |
| single_scan_floor_only_lidar_nodrift | 109 |
| single_scan_with_ceiling_lidar | 153 |
| single_scan_with_ceiling_lidar_nodrift | 145 |
| single_scan_with_ceiling_video | 155 |
| single_scan_with_ceiling_photo | 134 |
