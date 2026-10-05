"""Drift ablation: same capture, raw ARKit poses vs drift-corrected poses.

Metrics, all from the fused cloud (no ground truth needed):
- wall thickness: robust spread of wall-band points around each of the 12
  strongest wall planes. Drift fuses the same wall at two positions, which
  shows up as a thicker wall.
- floor residual: robust std of floor points about the floor plane
  (vertical drift splits the floor into layers).
- loop residual: ICP offset between revisited fragments before/after.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan2plan.layout import dominant_angle, Frame2D
from scan2plan.planes import detect_floor_ceiling


def wall_thickness(pts, w, floor_y, top, n_planes=12):
    h = pts[:, 1] - floor_y
    m = (h > 0.3) & (h < top) & (w >= 2)
    ang = dominant_angle(pts[m][:, [0, 2]], w[m])
    uv = Frame2D(ang, (0, 0), None).to_uv(pts[m])
    spreads = []
    for ax in (0, 1):
        x = uv[:, ax]
        hist, edges = np.histogram(x, bins=np.arange(x.min(), x.max() + 0.01, 0.01))
        used = np.zeros_like(hist, bool)
        for _ in range(n_planes // 2):
            k = int(np.argmax(np.where(used, -1, hist)))
            c = (edges[k] + edges[k + 1]) / 2
            used[max(k - 15, 0):k + 16] = True
            near = np.abs(x - c) < 0.10
            r = x[near] - np.median(x[near])
            spreads.append(1.4826 * np.median(np.abs(r)))
    return float(np.mean(spreads)), spreads


def main(name, out="out/bench"):
    rows = {}
    for mode in ("drift_off", "drift_on"):
        d = np.load(f"out/cache/{name}_{mode}_s3.npz", allow_pickle=True)
        pts, w, P, tag = d["pts"], d["w"], d["poses"], d["tag"].item()
        f, c = detect_floor_ceiling(pts, w, float(np.median(P[:, 1, 3])))
        top = (c["c"] - f["c"] - 0.15) if c else 2.4
        thick, _ = wall_thickness(pts, w, f["c"], top)
        plan = json.load(open(f"{out}/{name}{'' if mode == 'drift_on' else '_drift_off'}/plan.json"))
        rows[mode] = dict(wall_thickness_cm=round(100 * thick, 2), floor_residual_cm=round(100 * f["std"], 2),
                          footprint_m2=plan["property"]["footprint"]["value"], rooms=plan["property"]["room_count"],
                          loop_residual_mean_cm=round(100 * tag["loop_residual_after_m"]["mean"], 1) if tag.get("enabled") else None)
    on = json.load(open(f"{out}/{name}/plan.json"))["drift_correction"]
    rows["drift_off"]["loop_residual_mean_cm"] = round(100 * on["loop_residual_before_m"]["mean"], 1)
    rows["correction"] = {k: on[k] for k in ("method", "fragments", "loop_edges", "max_correction_m",
                                             "max_yaw_correction_deg", "heading_spread_deg")}
    print(name, json.dumps(rows, indent=1))
    json.dump(rows, open(f"{out}/ablation_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    # python scripts/ablation.py <out_dir> <capture_name>...
    for n in sys.argv[2:]:
        main(n, sys.argv[1])
