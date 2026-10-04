"""Score a photo/video-tier plan against the LiDAR-tier plan of the same space.

LiDAR is the reference here because the sample data has no tape ground
truth; its own error (~1-2 cm) is small next to the mono tiers' and is
included in the z-score denominator. Reports:
  - footprint relative error (photo gate: within +-8%)
  - wall length relative errors on matched walls (photo +-8%, video +-3%)
  - calibration: fraction of matched walls whose reference lies inside the
    tier's 95% interval (should be ~95%; much lower = overconfident)
  - stitch checks for the photo tier: room overlaps, adjacency recovered

    python scripts/compare_tiers.py <tier_plan.json> <lidar_plan.json> <out.json>
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repeatability import raster, register


def main(tier_path, ref_path, out_path):
    pt, pr = json.load(open(tier_path)), json.load(open(ref_path))
    R, t, err = register(pr, pt)  # maps tier plan into reference frame
    tf = lambda P: np.asarray(P) @ R.T + t
    allp = np.concatenate([np.array(r["polygon"]) for r in pr["rooms"]] + [tf(r["polygon"]) for r in pt["rooms"]])
    origin = allp.min(0) - 1
    shape = tuple((np.ceil((allp.max(0) + 1 - origin) / 0.05)).astype(int)[::-1])
    mt = {r["id"]: raster(tf(r["polygon"]), origin, shape) for r in pt["rooms"]}
    mr = {r["id"]: raster(r["polygon"], origin, shape) for r in pr["rooms"]}

    # Overlaps between tier rooms (photo gate: none).
    overlaps = []
    for a, b in combinations(pt["rooms"], 2):
        o = (mt[a["id"]] & mt[b["id"]]).sum() * 0.0025
        if o > 0.05:
            overlaps.append(dict(rooms=[a["id"], b["id"]], m2=round(float(o), 3)))

    walls = []
    for rt in pt["rooms"]:
        rr = max(pr["rooms"], key=lambda r: (mt[rt["id"]] & mr[r["id"]]).sum())
        if (mt[rt["id"]] & mr[rr["id"]]).sum() == 0:
            continue
        for wt in rt["walls"]:
            a0, a1 = tf(wt["start"]), tf(wt["end"])
            da = a1 - a0
            best = None
            for wr in rr["walls"]:
                b0, b1 = np.array(wr["start"]), np.array(wr["end"])
                db = b1 - b0
                cosang = abs(np.dot(da, db)) / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-9)
                dist = np.linalg.norm((a0 + a1) / 2 - (b0 + b1) / 2)
                if cosang > 0.95 and dist < 0.5 and (best is None or dist < best[0]):
                    best = (dist, wr)
            if best is None:
                continue
            wr = best[1]
            Lt, Lr = wt["length"]["value"], wr["length"]["value"]
            st, sr = wt["length"]["sigma"], wr["length"]["sigma"]
            if Lr < 0.5:
                continue
            lo, hi = wt["length"]["ci95"]
            walls.append(dict(tier_wall=wt["id"], ref_wall=wr["id"], tier=Lt, ref=Lr, rel_err=(Lt - Lr) / Lr,
                              z=(Lt - Lr) / np.hypot(st, sr), ref_in_ci95=bool(lo <= Lr <= hi)))
    ft, fr = pt["property"]["footprint"], pr["property"]["footprint"]
    rel = np.array([w["rel_err"] for w in walls]) if walls else np.array([])
    summary = dict(
        tier=pt["tier"], registration_median_residual_cm=round(100 * err, 1),
        footprint_tier=ft["value"], footprint_ref=fr["value"],
        footprint_rel_err=round((ft["value"] - fr["value"]) / fr["value"], 4),
        footprint_ref_in_ci95=bool(ft["ci95"][0] <= fr["value"] <= ft["ci95"][1]),
        matched_walls=len(walls),
        wall_abs_rel_err_median=round(float(np.median(np.abs(rel))), 4) if len(rel) else None,
        walls_within_8pct=round(float(np.mean(np.abs(rel) <= 0.08)), 3) if len(rel) else None,
        walls_within_3pct=round(float(np.mean(np.abs(rel) <= 0.03)), 3) if len(rel) else None,
        ci95_coverage=round(float(np.mean([w["ref_in_ci95"] for w in walls])), 3) if walls else None,
        z_rms=round(float(np.sqrt(np.mean([w["z"] ** 2 for w in walls]))), 2) if walls else None,
        rooms_tier=len(pt["rooms"]), rooms_ref=len(pr["rooms"]), room_overlaps=overlaps,
        adjacency_pairs=len(pt["adjacency"]))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(summary=summary, walls=walls), open(out_path, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
