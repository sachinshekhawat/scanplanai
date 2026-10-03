# Five tuning iterations (train / validation split)

`scripts/tune_layout.py` drives a model-training-style loop over the LiDAR layout stage: change one thing, score it on TRAIN, and keep it only if TRAIN improves. VALIDATION was scored once, at the end, for the starting and the final configuration. The full log is in `tuning_log.jsonl`.

- **TRAIN:** ARKitScenes visits 422382, 422803, 422813, 422826. That is 9 iPad Pro LiDAR walks with Faro laser GT, giving 35–42 laser-matched walls and 8–9 ceilings.
- **VALIDATION:** ARKitScenes visit 422386 (2 walks, never tuned on) and the sample-data cross-session pair (`single_scan_with_ceiling` vs `single_scan_floor_only`).
- **Score:** median |wall-length error| vs laser. It is reported with the share of walls ≤ 3%, interval coverage, and the number of matched walls, so a change cannot win by producing fewer, easier walls.

| it | change (root cause it targets) | TRAIN walls | median err | ≤ 3% | CI95 cov | decision |
|---|---|---|---|---|---|---|
| 0 | baseline (submitted version) | 35 | 15.6 cm | 26% | 80% | – |
| 1 | **see over furniture**: rays to wall points above ≈1.1 m also carve the floor under basins and shelves (`fusion.fan_high`, `rooms.USE_HIGH_FANS`) | 35 | **11.2 cm** | 31% | 80% | keep |
| 2 | merge "rooms" joined by an opening spanning ≥ 70% of a side (alcoves) | 35 | 11.2 cm | 31% | 80% | reject (no wall changed, lost a ceiling match) |
| 3 | prefer **full-height** faces (≥ 1.8 m) over clutter fronts when snapping (`PREFER_TALL`, 3 cm penalty; 3 values tried) | 35 | **10.9 cm** | 31% | 80% | keep |
| 4 | structural-wall threshold × jog threshold grid (11 settings): `WALL_MIN_COV` 0.6→**0.7**, `MIN_EDGE` 0.40→**0.25** | 41 | **7.2 cm** | **32%** | 93% | keep (best median with the most walls; 0.8 gave 5.5 cm but on 30 walls) |
| 5 | recalibrate room-boundary σ on TRAIN (smallest giving ≥ 95%): `SIGMA_LAYOUT` 3.5→**4.75 cm** | 41 | 7.2 cm | 32% | **95%** | keep |

TRAIN total: median wall error **15.6 → 7.2 cm** (−54%), walls within 3% **26 → 32%**, coverage **80 → 95%**. Ceilings: median 0.6–0.8 cm on 8–9 rooms.

## Validation: scored once, baseline vs final
| set | metric | baseline | final |
|---|---|---|---|
| ARKitScenes held-out 422386 | laser-matched walls | 1 | **7** |
| | median wall error | 23.8 cm | **16.7 cm** |
| | wall CI95 coverage | 100% (n = 1) | **100%** (n = 7) |
| | ceiling median error / coverage | 1.4 cm / 50% | **0.5 cm / 100%** |
| sample-data cross-session pair | median wall difference | 10.3 cm | **7.2 cm** |
| | CI95 coverage of differences | 91% | **100%** |
| | walls matched / within 1 cm | 22 / 23% | 16 / 12% |
| | footprint IoU | 0.848 | 0.804 |

Reading:
- **Held-out lab data improves on every metric.** The test is small (7 walls), but it is real laser ground truth.
- **On the sample flat:** errors shrink and the intervals now always contain the truth. But fewer walls are matched across sessions, so the strict 1 cm repeatability gate looks worse (12% vs 23%). We shipped the train-selected configuration anyway; choosing per validation result would turn validation into a second train set.
- **Remaining gap to the gates:** 1 cm / 3% wall accuracy needs a layout model that is not driven by free space (e.g. explicit wall-plane graph optimisation). Noise in the measured planes (1–2 cm vs laser) is not the bottleneck.
