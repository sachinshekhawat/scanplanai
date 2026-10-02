# Device matrix

| Tier | Hardware | Capture app | Pipeline runtime (M1 laptop, 16 GB) | Accuracy it honestly delivers (see benchmark) |
|---|---|---|---|---|
| LiDAR | iPhone 12 Pro or later Pro/Pro Max, iPad Pro (2020+) | Stray Scanner (free) | 40 s per room; 2–3 min for a 100 m walk of a 6-room flat (CPU only) | Wall faces σ ≈ 0.4–1 cm; opening widths σ ≈ 1 cm with both jambs found; ceiling σ ≈ 1 cm when scanned. Cross-session repeatability: see `benchmark/results/*/report.md` |
| Video | Any iPhone 15 or later (no LiDAR needed) | Native Camera, 1x, landscape | 5–15 min per walkthrough (SfM + DA3 depth on Apple GPU via MPS) | Metric scale from monocular depth: 4% σ per capture (calibrated against LiDAR); wall lengths widen to ±6–15 cm at 95% |
| Photo | Any iPhone 15 or later | Native Camera, 2–8 stills per room | 1–2 min per room | Monocular scale 6.5% σ per room, plus 5 cm placement σ per room in the stitch; rooms whose photos do not register degrade to fewer views and wider intervals |

Notes
- Every tier writes the same `plan.json` contract. Intervals widen with the tier's error model (`src/scan2plan/measure.py`, `_video_error_model`, `PHOTO_SCALE_REL`), and quantities that were not observed are marked `observed: false`.
- Non-Pro iPhones have no LiDAR, so Stray Scanner records no depth on them. The protocol sends those users to the video or photo tier.
- Weights stay on the local machine. Nothing calls our infrastructure.
