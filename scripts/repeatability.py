"""Repeatability: two captures of the same space at the same tier.

The two plans live in different frames (each capture starts its own ARKit
world). They are registered with 2D ICP on wall samples (four 90-degree
start hypotheses, since the dominant-axis frame is only defined mod 90),
rooms are matched by overlap, and walls are matched by position and
orientation inside matched rooms. Gate: |dL| <= 1 cm or <= 0.5% of L.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree


def wall_samples(plan, step=0.05):
    pts = []
    for r in plan["rooms"]:
        for w in r["walls"]:
            if not w["observed"]:
                continue
            a, b = np.array(w["start"]), np.array(w["end"])
            n = max(int(np.linalg.norm(b - a) / step), 1)
            pts.append(a + (b - a) * np.linspace(0, 1, n + 1)[:, None])
    return np.concatenate(pts)


def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def icp2d(src, dst, R, t, iters=60):
    tree = cKDTree(dst)
    for _ in range(iters):
        s = src @ R.T + t
        d, j = tree.query(s)
        m = d < np.percentile(d, 80)
        A, B = s[m], dst[j[m]]
        ca, cb = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
        dR = (U @ Vt).T
        if np.linalg.det(dR) < 0:
            Vt[-1] *= -1
            dR = (U @ Vt).T
        R, t = dR @ R, dR @ (t - ca) + cb
    d, _ = tree.query(src @ R.T + t)
    return R, t, float(np.median(d))


def register(pa, pb):
    A, B = wall_samples(pa), wall_samples(pb)
    best = None
    for k in range(4):
        R0 = rot(k * np.pi / 2)
        t0 = A.mean(0) - B.mean(0) @ R0.T
        R, t, err = icp2d(B, A, R0, t0)
        if best is None or err < best[2]:
            best = (R, t, err)
    return best


def raster(poly, origin, shape, cell=0.05):
    img = np.zeros(shape, np.uint8)
    P = ((np.asarray(poly) - origin) / cell).astype(np.int32)
    cv2.fillPoly(img, [P], 1)
    return img.astype(bool)


def main(a_path, b_path, out_path):
    pa, pb = json.load(open(a_path)), json.load(open(b_path))
    R, t, err = register(pa, pb)
    tf = lambda P: np.asarray(P) @ R.T + t
    allp = np.concatenate([np.array(r["polygon"]) for r in pa["rooms"]] + [tf(r["polygon"]) for r in pb["rooms"]])
    origin = allp.min(0) - 1
    shape = tuple((np.ceil((allp.max(0) + 1 - origin) / 0.05)).astype(int)[::-1])
    ma = {r["id"]: raster(r["polygon"], origin, shape) for r in pa["rooms"]}
    mb = {r["id"]: raster(tf(r["polygon"]), origin, shape) for r in pb["rooms"]}
    rows = []
    room_rows = []
    for ra in pa["rooms"]:
        best = max(pb["rooms"], key=lambda rb: (ma[ra["id"]] & mb[rb["id"]]).sum())
        inter = (ma[ra["id"]] & mb[best["id"]]).sum()
        iou = inter / max((ma[ra["id"]] | mb[best["id"]]).sum(), 1)
        if iou < 0.5:
            continue
        room_rows.append(dict(room_a=ra["id"], room_b=best["id"], iou=round(float(iou), 3),
                              area_a=ra["floor_area"]["value"], area_b=best["floor_area"]["value"],
                              ceiling_a=ra["ceiling_height"]["value"], ceiling_b=best["ceiling_height"]["value"]))
        for wa in ra["walls"]:
            a0, a1 = np.array(wa["start"]), np.array(wa["end"])
            da = a1 - a0
            cand = []
            for wb in best["walls"]:
                b0, b1 = tf(wb["start"]), tf(wb["end"])
                db = b1 - b0
                cosang = abs(np.dot(da, db)) / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-9)
                dist = np.linalg.norm((a0 + a1) / 2 - (b0 + b1) / 2)
                if cosang > 0.95 and dist < 0.3:
                    cand.append((dist, wb))
            if not cand:
                continue
            _, wb = min(cand, key=lambda c: c[0])
            La, Lb = wa["length"]["value"], wb["length"]["value"]
            dL = abs(La - Lb)
            rows.append(dict(room=ra["id"], wall_a=wa["id"], wall_b=wb["id"], length_a=La, length_b=Lb,
                             diff_cm=round(100 * dL, 2), diff_pct=round(100 * dL / max(La, 1e-9), 2),
                             both_observed=wa["observed"] and wb["observed"],
                             passes=bool(dL <= 0.01 or dL <= 0.005 * La)))
    obs = [r for r in rows if r["both_observed"] and r["length_a"] >= 0.5]
    summary = dict(registration_median_residual_cm=round(100 * err, 2), matched_rooms=len(room_rows),
                   matched_walls=len(rows), walls_scored=len(obs),
                   pass_rate=round(float(np.mean([r["passes"] for r in obs])), 3) if obs else None,
                   median_diff_cm=round(float(np.median([r["diff_cm"] for r in obs])), 2) if obs else None,
                   p90_diff_cm=round(float(np.percentile([r["diff_cm"] for r in obs], 90)), 2) if obs else None)
    out = dict(a=str(a_path), b=str(b_path), summary=summary, rooms=room_rows, walls=rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    for r in room_rows:
        print(" ", r)
    for r in obs:
        print(f"  {r['wall_a']:>8} {r['length_a']:.3f} vs {r['length_b']:.3f}  diff {r['diff_cm']:5.1f} cm {r['diff_pct']:5.1f}%  {'PASS' if r['passes'] else 'fail'}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
