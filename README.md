# scan2plan

Phone capture in, dimensioned and stitched floor plan out — from LiDAR, from a
handheld video, or from 2–8 photos per room — with a 95% interval on every
measurement, damage regions on surfaces, concealed-damage flags and scope
line items. One command per capture.

```
scan2plan run lidar  <stray_scanner_folder>
scan2plan run video  <clip.mov>
scan2plan run photos <folder_of_room_folders>
```

Output: `out/<name>_<tier>/plan.json` (schema: `schema/plan.schema.json`) and
`plan.png` (rendered plan).

## Setup (clean macOS / Linux machine, ~10 minutes)

Needs Python 3.10–3.12 and ~3 GB of disk for wheels and weights. No GPU needed;
Apple-silicon MPS is used when present.

```bash
git clone <this repo> && cd scan2plan
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .     # ~4 min
scripts/fetch_weights.sh                                 # ~3 min, ~1.5 GB, then works offline
```

Run on a fresh capture (see `docs/CAPTURE_PROTOCOL.md` for how to capture):

```bash
scan2plan run lidar ~/Downloads/2026-10-05_120000      # unzipped Stray Scanner folder
```

## Reproducing every reported number

```bash
scripts/run_benchmark.sh <dir with the three sample captures> out/bench
```

regenerates the LiDAR-tier plans, the repeatability table and the drift
ablation inputs. Mono tiers, photo sets and the depth calibration have their
own scripts listed in `docs/REPORT.md` (Reproduction). The fix loop:
`git checkout fixloop-before && scripts/run_benchmark.sh .. out/before`, then
the same on `main` into `out/after`.

## How it works (short)

| Stage | LiDAR | Video | Photos |
|---|---|---|---|
| Poses | ARKit (Stray) + drift correction: fragment ICP loop closure and Manhattan heading prior in a pose graph | COLMAP SfM on DISK + LightGlue matches | same, all pairs across all room folders (doorway photos stitch rooms) |
| Depth | LiDAR, high-confidence pixels | Depth Anything V2 Metric-Indoor, bias-calibrated against LiDAR, per-image scale fixed by SfM points | same |
| Up / floor | ARKit gravity | level-camera-axis gravity, refined by floor plane | same |
| Rooms | door-gap bridging on tall-surface wall map | same | one room per folder |
| Walls | global wall planes; room outlines snap to them | same | same |
| Errors | `errors.LIDAR` | `errors.VIDEO` + scale term | `errors.PHOTO` + scale term |

Damage: OWLv2 zero-shot detection on upright keyframes → back-projected to the
surface it lies on → metric extent; rules `CD-01..04` for concealed damage;
scope items keyed to wall/floor/ceiling ids.

## Repository map

```
scan2plan/io.py          Stray Scanner loader
scan2plan/cloud.py       depth fusion
scan2plan/drift.py       loop closure + Manhattan heading pose graph
scan2plan/planes.py      floor / ceiling
scan2plan/layout.py      wall map, dominant axes, room segmentation
scan2plan/walls.py       global wall planes
scan2plan/rooms.py       per-room outline, walls, openings, heights
scan2plan/errors.py      per-tier error model
scan2plan/pipeline.py    shared analysis + JSON + render
scan2plan/mono/          video/photo tiers (frames, features, sfm, depth, build, run)
scan2plan/damage/        detection, surface location, rules + scope
scripts/                 benchmark, repeatability, ablation, calibration, photo-set tools
docs/                    capture protocol, device matrix, fix declaration, report, compliance
```

## Models and data used (disclosure)

- Depth Anything V2 Metric-Indoor Base/Small (Hugging Face `depth-anything/...`), Apache-2.0 / CC-BY-NC-4.0 for Base — see model cards.
- DISK + LightGlue via kornia (Apache-2.0 code; weights from the authors).
- OWLv2 `google/owlv2-base-patch16-ensemble` (Apache-2.0).
- COLMAP via pycolmap (BSD).
- No external API is called at run time; weights are fetched once by script.
