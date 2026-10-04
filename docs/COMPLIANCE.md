# Compliance matrix

Status: ✅ done and verified on the sample data · 🟡 implemented, partially
verified or below the gate · ❌ not done (reason given). Gates are the case
study's; "sample" = the three Stray Scanner captures provided by email.

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| 1.1 | Capture route (Route 2: stock apps) + one-page protocol | `docs/CAPTURE_PROTOCOL.md` | protocol | ✅ |
| 1.2 | Photo tier: 2–8 stills/room, folders → stitched plan | `scan2plan/mono/run.py`, `mono/depthgraph.py` | `out/bench/photos/plan.json` | 🟡 runs end to end; places only rooms whose photos link by matches |
| 1.3 | Video tier: handheld clip | `scan2plan/mono/run.py`, `mono/sfm.py`, `mono/build.py` | `out/bench/video/plan.json` | 🟡 runs end to end; covers part of the apartment on the sample clip |
| 1.4 | LiDAR tier: depth, poses, intrinsics | `scan2plan/pipeline.py` | `out/bench/<capture>/plan.json` | ✅ |
| 1.5 | Device matrix with honest accuracy | `docs/DEVICE_MATRIX.md`, `docs/REPORT.md` §Benchmark | table | ✅ |
| 2.1 | Per-room plan: walls, ceiling height, floor area, openings | `scan2plan/rooms.py` | `rooms[]` in plan.json | ✅ |
| 2.2 | Stitched multi-room plan with adjacency | `scan2plan/layout.py`, `pipeline.py` | `adjacency[]`, `plan.png` | ✅ LiDAR · 🟡 mono |
| 2.3 | Damage regions: class + metric extent per surface | `scan2plan/damage/` | `damage_regions[]` | 🟡 implemented; no staged damage in sample → true-positive rate unmeasured; false positives on sample reported |
| 2.4 | Concealed-damage flags with rule that fired | `scan2plan/damage/rules.py` | `concealed_damage_flags[]` | 🟡 rules CD-01..04 implemented; never fired on sample (no damage) |
| 2.5 | Scope line items keyed to surfaces | `scan2plan/damage/rules.py` | `scope_items[]` | 🟡 as 2.4 |
| 2.6 | Confidence interval on every measurement | `scan2plan/errors.py` | `{value, sigma, ci95}` | ✅ |
| 2.7 | One command per capture | `scan2plan/__main__.py` | `scan2plan run <tier> <input>` | ✅ |
| 2.8 | JSON to published schema | `schema/plan.schema.json` | validated in benchmark | ✅ (brief's schema not supplied; ours published) |
| 2.9 | Rendered plan | `scan2plan/render.py` | `plan.png` | ✅ |
| 2.10 | Benchmark: multi-room (3+ rooms + connector) | sample `single_scan_*` | | ✅ (sample) |
| 2.11 | Benchmark: furnished room with staged damage, 2 classes | | | ❌ not in sample data; needs a new capture |
| 2.12 | Benchmark: same rooms at all three tiers | `scripts/make_photo_set.py` | photo/video derived from the LiDAR captures' RGB | 🟡 photo/video inputs are derived from the same walk, not separate captures |
| 2.13 | Same room twice at one tier (repeatability) | `scripts/repeatability.py` | `out/bench/repeatability.json` | 🟡 measured on the two apartment captures; gate fails (see fix loop) |
| 2.14 | Laser/tape ground truth | | | ❌ none for sample data; LiDAR used as reference for mono tiers |
| 2.G1 | Opening widths ≤ 2 cm on ≥ 85% | `rooms.find_openings` | | 🟡 measured, unscored (no GT) |
| 2.G2 | Ceiling height ≤ 1.5 cm, spread ≤ 1 cm | `rooms.measure_room` | | 🟡 with-ceiling capture only; the floor-only capture has no ceiling to compare |
| 2.G3 | Repeatability ≤ 1 cm or 0.5% per wall | `scripts/repeatability.py` | | ❌ fails (0%); fix loop moved median 16.0 → 12.9 cm |
| 2.G4 | Drift accountability + ablation | `scan2plan/drift.py`, `scripts/ablation.py` | `out/bench/ablation_*.json` | ✅ |
| 2.G5 | Photo-tier whole-property stitch | `mono/depthgraph.py` | | 🟡 see report |
| 3 | Head-to-head vs consumer app on 2 rooms | | | ❌ no consumer-app export of the sample rooms exists; requires recapturing the same rooms |
| 4 | Fix loop: declaration, before/after, diff | `docs/FIX_DECLARATION.md`, tag `fixloop-before` | `out/before`, `out/after` | ✅ shipped; gate not passed; post-mortem in report |
| 5 | Process evidence | git history | | ✅ |
| D3 | README to running in < 15 min | `README.md` | | ✅ |
| D4 | Reproduction bundle | `scripts/run_benchmark.sh`, `scripts/*` | | ✅ |
| D5 | Benchmark report | `docs/REPORT.md` §Benchmark | | ✅ |
| D7 | Technical report ≤ 6 pages | `docs/REPORT.md` | | ✅ |
| D8 | Raw benchmark data | sample captures (not in repo: 900 MB) + `out/` regenerated | | 🟡 inputs are the provided sample; no extra captures |
| C1 | Mirrors, glass, wet surfaces, low light covered | `docs/REPORT.md` §Failure modes, protocol "Avoid" | | 🟡 documented and mitigated, not benchmarked |
