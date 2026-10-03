# Tuning iterations (train / validation split): what happened

`scripts/tune_layout.py` drives the loop; every scored run is in `tuning_log.jsonl`.
- **TRAIN:** ARKitScenes visits 422382, 422803, 422813, 422826. 9 iPad Pro LiDAR walks, each with a Faro laser scan.
- **VALIDATION:** visit 422386 (never tuned on) and the sample-data cross-session pair.

## Iterations
| it | change | TRAIN result | decision |
|---|---|---|---|
| 1 | carve free space also from rays to points above 1.1 m ("see over furniture") | median wall err 15.6 → 11.2 cm | kept at first |
| 2 | merge rooms joined by an opening spanning ≥ 70% of a side (alcoves) | no change | rejected |
| 3 | prefer full-height faces when snapping room edges | 11.2 → 10.9 cm | kept at first |
| 4 | stricter structural-wall coverage 0.6 → 0.7, finer jog threshold 0.4 → 0.25 m | 10.9 → **7.2 cm**, 41 walls | kept at first |
| 5 | refit room-boundary σ on TRAIN | coverage 93 → 95% | kept at first |
| 6 | **check on the sample flat (product surface):** iterations 1–5 merged rooms. single_room went from 3 rooms to 1, the multi-room scans from 6/7 to 5/5, and the drift ablation flipped (on 0.804 vs off 0.817) | — | regression found |
| 7 | root cause: coverage 0.7 drops door-jamb wall pixels, so doorways stop sealing. Tried separate thresholds for sealing (0.6) vs boundaries (0.7/0.8); rooms come back but the TRAIN gain disappears (15.6 cm) | 15.6 cm | rejected |
| 8 | **scoring audit:** the 6 worst TRAIN "errors" (±34–39 cm) were walls the laser scan does not cover. `gt_face` latched onto unrelated points. Added a coverage gate (laser must span ≥ 40% of the wall) | 24 valid walls | scoring fix kept |

With valid ground truth and the partition preserved, **none of the layout changes measurably improves wall accuracy**:

| config (TRAIN, 24 GT-valid walls) | median wall err | ≤ 3% | CI95 cov |
|---|---|---|---|
| submitted layout | 10.6 cm | 25% | 83% |
| + see-over-furniture | 10.6 cm | 25% | 83% |
| + prefer full-height faces | 10.9 cm | 25% | 83% |
| + jog 0.25 m | 10.9 cm | 22% | 85% |

**Shipped:** the submitted layout, unchanged. The new code paths stay behind flags set to off (`USE_HIGH_FANS`, `PREFER_TALL`, `ALCOVE_FRAC = 1.0`), and the GT coverage gate is kept in the scorer.

## Final scores (submitted layout + laser-fitted scale + 3.5 cm boundary σ)
| set | walls / rooms | wall median err | wall CI95 cov | ceiling median err | ceiling CI95 cov |
|---|---|---|---|---|---|
| TRAIN (9 walks) | 24 walls | 10.6 cm | 83% | 0.75 cm (7 of 9 rooms ≤ 1.8 cm) | 78% |
| VAL ARKitScenes 422386 | 1 wall, 2 rooms | 23.8 cm | 100% | 1.4 cm | 50% |
| VAL sample pair (2 sessions) | 22 walls | 10.3 cm difference | **91%** | – | – |

The two ceiling misses on TRAIN (+12.9 and −10.7 cm) are rooms where the laser "ceiling" slab picked a shelf or soffit; the other 7 are within 1.8 cm. Inflating σ to 12 cm per face would reach 95% wall coverage, but intervals that wide are useless, so we did not.

## Lessons
1. **A train-split gain is not enough; a product-level guard is required.** The 15.6 → 7.2 cm improvement came from merging rooms (fewer, larger polygons whose outer walls happen to be laser-covered) and from invalid GT walls. Room count on the sample flat is now part of the acceptance check (`/tmp/roomcount.py` equivalent in `tune_layout.py val`).
2. **The ground-truth extractor needs as much scrutiny as the method.** Six of the "worst errors" were scoring bugs.
3. **Wall accuracy (≈ 10 cm) is limited by layout, not sensing.** The measured planes agree with the laser to 1–2 cm. The next real step is a different layout model, such as wall-plane graph optimisation with per-room closure constraints, not more threshold tuning.
