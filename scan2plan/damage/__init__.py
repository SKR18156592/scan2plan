"""Damage stage shared by all tiers: detect (subprocess) -> locate -> rules."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .locate import locate
from .rules import concealed_flags, scope_items


def frames_from_video(video, indices, out_dir):
    """Write the given frame indices of a video as JPEGs; returns paths."""
    import cv2
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    want, paths = set(indices), {}
    v = cv2.VideoCapture(str(video))
    k = 0
    while want:
        ok, img = v.read()
        if not ok:
            break
        if k in want:
            p = out_dir / f"{k:06d}.jpg"
            cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            paths[k] = str(p)
            want.discard(k)
        k += 1
    return paths


def run(cap, plan, image_paths, rot, work, prof, min_views):
    """image_paths: {frame index in cap: jpeg path}. Returns (regions, flags, items, info)."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    fj, dj = work / "damage_frames.json", work / "damage_detections.json"
    json.dump(dict(images=[dict(id=int(i), path=p, rot=int(rot)) for i, p in sorted(image_paths.items())]), open(fj, "w"))
    if not dj.exists():
        with open(work / "damage.log", "w") as log:
            r = subprocess.run([sys.executable, "-m", "scan2plan.damage.detect", str(fj), str(dj)], stdout=log, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            return [], [], [], dict(status="detector_failed", log=str(work / "damage.log"))
    det = json.load(open(dj))
    regions = locate(det["detections"], cap, plan, min_views=min_views)
    for k, r in enumerate(regions):
        r["id"] = f"d{k + 1}"
        a, b = r["extent"]
        # Extent sigma: box looseness (~10% of extent) plus surface position.
        sa, sb = 0.1 * a + prof.surface_bias, 0.1 * b + prof.surface_bias
        r["extent_sigma"] = [round(sa, 4), round(sb, 4)]
        r["area"] = dict(value=round(a * b, 4), sigma=round(float(np.hypot(sa * b, sb * a)), 4))
        r["area"]["ci95"] = [round(r["area"]["value"] - 1.96 * r["area"]["sigma"], 4),
                             round(r["area"]["value"] + 1.96 * r["area"]["sigma"], 4)]
    flags = concealed_flags(regions)
    items = scope_items(regions, plan)
    info = dict(status="ok", model=det["model"], threshold=det["threshold"], frames_scanned=len(image_paths),
                raw_detections=len(det["detections"]), min_views=min_views)
    return regions, flags, items, info
