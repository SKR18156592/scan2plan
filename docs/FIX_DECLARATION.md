# Fix declaration

Declared before the fix was written. Before-run: git tag `fixloop-before`.

## 1. Worst-performing gate

**Repeatability** (two captures of the same space at the same tier agree
within 1 cm or 0.5% per wall).

The sample data has the same apartment captured twice at the LiDAR tier
(`single_scan_floor_only`, `single_scan_with_ceiling`). It is the only gate
scorable without tape/laser ground truth, and it fails worst:

| metric (walls observed in both, length >= 0.5 m) | before |
|---|---|
| walls passing (<= 1 cm or <= 0.5%) | **0 / 8 (0%)** |
| median length difference | **17.5 cm** |
| p90 length difference | 102.7 cm |
| rooms matched (IoU >= 0.5) | 4 of 7 |

Regenerate: `git checkout fixloop-before && scripts/run_benchmark.sh .. out/before`.

## 2. Root-cause hypothesis

Wall lengths are taken between vertices of a room polygon traced from the
free-space mask. The mask's extent depends on what a capture happened to
see: floor hidden under furniture, a doorway leaking into the next room,
clutter along a wall. So the *outline* differs between captures even
though the wall *surfaces* are the same physical planes.

Evidence:
- Registered overlay (`out/repeatability/overlay.png`): where both captures
  draw a wall, the lines coincide to a few cm; disagreements are rooms that
  stop short of a wall (floor_only `r2` is ~1.2 m shorter than the same
  room in with_ceiling) or extra jogs (`r1_w6`: 1.15 vs 2.01 m).
- Per-wall surface refinement already lands on the wall (median surface
  spread 1.3 cm), so the error is in *which* surfaces bound the room, not
  in locating a surface.
- The two captures' floor areas for the same apartment agree within 2%
  (52.8 vs 53.2 m²) while individual rooms differ by up to 35%: the
  error is in partitioning and outline, not in global geometry.

## 3. Fix

Detect wall planes once per capture, globally (long vertical planes in the
axis-aligned frame, fitted on all their points), and build each room's
outline from that plane arrangement: every mask edge snaps to the nearest
supported wall plane on its outer side (search up to 0.6 m), and edges that
snap to the same plane merge. Wall length then becomes a plane-to-plane
distance, which does not depend on how much floor the capture saw.

## Predicted result

- median length difference: **17.5 cm -> <= 3 cm**
- pass rate (1 cm / 0.5%): **0% -> 30-50%**. The gate is tighter than
  LiDAR surface bias (~1 cm per surface, two surfaces per length), so
  full pass is not expected from this fix alone.
- rooms matched: 4 -> >= 5


---

## Result (added after shipping; the declaration above is unchanged)

| (harness v2, rotation locked to k·90°) | before | after |
|---|---|---|
| walls passing | 0 / 9 | 0 / 12 |
| median difference | 16.0 cm | 12.9 cm |
| p90 difference | 97.3 cm | 29.0 cm |
| rooms matched | 4 | 5 |

The prediction (median <= 3 cm, 30-50% passing) was **badly wrong**. The
gate did not move from fail. Post-mortem in `docs/REPORT.md` §7: the harness
let rotation float (fixed: before-number restated 17.5 -> 16.0 cm under the
corrected harness), a second root cause (~5 deg heading drift) was found and
fixed by a Manhattan heading prior, and the declared root cause (outline
from free-space masks) is only partly addressed by plane snapping because
the two captures still partition the apartment differently.
