# Device matrix

Which tier runs on which phone, what the capture needs, and what accuracy each
tier honestly delivers. "Measured" numbers come from the sample captures
(`docs/REPORT.md`, Benchmark); there is no tape ground truth for them, so the
LiDAR tier is scored by repeatability and the mono tiers against the LiDAR
plan of the same space. Numbers marked *prior* are not yet measured.

| Device | Photos (tier 1) | Video (tier 2) | LiDAR (tier 3) |
|---|---|---|---|
| iPhone 15, 15 Plus, 16, 16 Plus, 16e, 17, Air | ✅ Camera app | ✅ Camera app | ❌ no LiDAR sensor |
| iPhone 15 Pro / Pro Max, 16 Pro / Pro Max, 17 Pro / Pro Max | ✅ | ✅ | ✅ Stray Scanner |
| iPhone 12 Pro – 14 Pro (outside the brief's "15 or newer", works) | ✅ | ✅ | ✅ |
| iPad Pro (LiDAR models) | ✅ | ✅ | ✅ |

## What each tier delivers

| | Photos | Video | LiDAR |
|---|---|---|---|
| Input | 2–8 upright photos per room, one folder per room | one walkthrough clip | Stray Scanner folder (RGB, depth, confidence, ARKit poses) |
| Poses | depth-lifted match graph (sim3) | COLMAP SfM on DISK+LightGlue, sub-models chained by depth | ARKit + loop closure + Manhattan heading |
| Metric scale | Depth Anything V2 Metric-Indoor, bias-calibrated against LiDAR | same | LiDAR |
| Ceiling height | only if ceiling visible in photos | if the clip sweeps the ceiling | yes (protocol sweeps it) |
| Interval model (1σ) | surface 15 cm + 50% of length (calibrated) | surface 10 cm + 45% (calibrated) | surface 1 cm + 0.5% (prior) |
| Measured on sample | 27/40 photos placed; walls off by median 86% vs LiDAR; 33% interval coverage | 100/229 frames placed; walls off by median 51%; 83% coverage | repeatability median 12.9 cm between two captures; ceiling ±4 cm (95%) |
| Processing time (M1, 8 GB) | ~40 min for 40 photos (memory-bound; 16 GB recommended) | ~30 min for a 2 min clip | 1–7 min |

Measured accuracy per tier is filled in from the benchmark: see the table in
`docs/REPORT.md` §Benchmark (kept in one place so the numbers cannot drift
between documents).
