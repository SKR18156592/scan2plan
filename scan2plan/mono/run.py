"""Photo and video tiers, end to end.

    python -m scan2plan.mono.run video <clip.mp4|.mov> --out DIR
    python -m scan2plan.mono.run photos <dir with one sub-folder per room> --out DIR

Stages run as separate processes (pycolmap and torch cannot share one):
frames -> DISK+LightGlue features -> COLMAP SfM -> orientation -> metric
depth -> scale/gravity alignment -> fusion -> shared analysis (pipeline.py).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .. import errors
from ..cloud import fuse
from ..layout import CELL
from ..pipeline import analyze

# Depth Anything V2 Metric-Indoor over-estimates depth on iPhone imagery.
# Measured against LiDAR depth on the same frames (scripts/eval_mono_depth.py):
# single_room 1.341, single_scan_floor_only 1.286 (fit, mean 1.313);
# single_scan_with_ceiling 1.392 (held out: +6.0% residual). The held-out
# residual is why the mono profiles carry a 4-5% scale sigma on top.
MONO_DEPTH_BIAS = 1.313


def _stage(args, log):
    t = time.time()
    with open(log, "a") as f:
        r = subprocess.run([sys.executable, "-m", *args], stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"stage failed: {' '.join(args)} (see {log})")
    return time.time() - t


def prepare(kind, src, work):
    """Lay out <work>/images (video keyframes or a copy of the photo folders)."""
    work = Path(work)
    img = work / "images"
    if img.exists() and any(img.rglob("*.jpg")):
        return img
    img.mkdir(parents=True, exist_ok=True)
    if kind == "video":
        from .frames import extract
        extract(src, img, every_s=0.5)
    else:
        from PIL import Image, ImageOps
        for room in sorted(p for p in Path(src).iterdir() if p.is_dir()):
            for p in sorted(room.iterdir()):
                if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".heic"):
                    continue
                dst = img / room.name / (p.stem + ".jpg")
                dst.parent.mkdir(parents=True, exist_ok=True)
                im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
                # Keep EXIF focal information out of the copy: the copy is
                # already upright and the SfM focal prior covers iPhones.
                im.thumbnail((2016, 2016))
                im.save(dst, quality=92)
    return img


def run(kind, src, out, model="base"):
    t0 = time.time()
    out = Path(out)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    log = work / "stages.log"
    timing = {}
    img = prepare(kind, src, work)
    timing["prepare_s"] = time.time() - t0
    feats = work / "features.npz"
    if not feats.exists():
        args = ["scan2plan.mono.features", str(img), str(feats)]
        if kind == "video":
            args += ["--sequential", "6"]
        timing["features_s"] = _stage(args, log)
    sfm_dir = work / "sfm"
    if not (sfm_dir / "sfm.json").exists():
        args = ["scan2plan.mono.sfm", str(img), str(sfm_dir), "--features", str(feats)]
        if kind == "video":
            args.append("--sequential")
        timing["sfm_s"] = _stage(args, log)

    from .build import load, orientation, refine_gravity_with_floor
    rot = orientation(sfm_dir / "sfm.json")
    depth_dir = work / f"depth_{model}"
    if not depth_dir.exists():
        timing["depth_s"] = _stage(["scan2plan.mono.depth", str(img), str(depth_dir), "--model", model, "--rot", str(rot)], log)

    sfm = json.load(open(sfm_dir / "sfm.json"))
    n_total = len(sfm["images"])
    reg = [len(m["images"]) for m in sfm["models"]]
    cap, stats = load(sfm_dir / "sfm.json", depth_dir, image_root=img)
    # Undo the network's measured scale bias (both the SfM scale and every
    # depth map inherit it).
    cap.poses[:, :3, 3] /= MONO_DEPTH_BIAS
    cap.depth_scale = cap.depth_scale / MONO_DEPTH_BIAS
    stats.update(image_rotation_quarter_turns=rot, depth_model=model, depth_bias_correction=MONO_DEPTH_BIAS,
                 models_registered=reg, images_total=n_total)

    t = time.time()
    pts, w = fuse(cap, step=1, max_depth=5.0)
    cap, R, pts = refine_gravity_with_floor(cap, pts)
    timing["fuse_s"] = time.time() - t

    prof = errors.VIDEO if kind == "video" else errors.PHOTO
    # Relative scale uncertainty: held-out bias residual (6%) shrinks with
    # the number of images averaged, floor at the profile's scale sigma.
    extra = float(np.hypot(stats["scale_sigma_rel"], 0.06 / np.sqrt(max(len(cap) / 20, 1))))

    room_labels = None
    if kind == "photos":
        room_labels = _folder_rooms(cap, pts)
    tag = dict(enabled=False, method="COLMAP global bundle adjustment (no separate pose graph)")
    title = f"{Path(src).name} - {kind} tier ({len(cap)}/{n_total} images registered)"
    return analyze(Path(src).stem, "photo" if kind == "photos" else "video", pts, w, cap.poses, tag, timing, t0,
                   out, prof, extra_scale=extra, room_labels=room_labels, meta=stats, title=title)


def _folder_rooms(cap, pts_unused):
    """Photo tier partition: each folder is one room. A grid cell belongs to
    the room whose images saw the most floor there; cells nobody saw are
    left out. This is what keeps stitched rooms from overlapping."""
    from scipy import ndimage as ndi
    rooms = sorted(set(cap.room_of))

    def labels(frame, g, free):
        votes = np.zeros((len(rooms),) + frame.shape, np.float32)
        from ..io import backproject
        for i in range(len(cap)):
            p = backproject(cap, i, max_depth=5.0)
            cc = frame.to_cell(frame.to_uv(p))
            ok = (cc[:, 0] >= 0) & (cc[:, 0] < frame.shape[1]) & (cc[:, 1] >= 0) & (cc[:, 1] < frame.shape[0])
            np.add.at(votes[rooms.index(cap.room_of[i])], (cc[ok, 1], cc[ok, 0]), 1.0)
        best = votes.argmax(0) + 1
        lab = np.where(free & (votes.max(0) > 0), best, 0)
        # One connected piece per room: keep each room's largest component.
        out = np.zeros_like(lab)
        for k in range(1, len(rooms) + 1):
            comp, n = ndi.label(lab == k)
            if n:
                sizes = ndi.sum(np.ones_like(comp), comp, range(1, n + 1))
                out[comp == 1 + int(np.argmax(sizes))] = k
        return out

    return labels


def main():
    ap = argparse.ArgumentParser(prog="scan2plan.mono.run")
    ap.add_argument("kind", choices=["video", "photos"])
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="base", choices=["small", "base"])
    a = ap.parse_args()
    r = run(a.kind, a.src, a.out, a.model)
    print(f"{r['property']['room_count']} rooms, footprint {r['property']['footprint']['value']:.2f} m², "
          f"{r['timing_s']['total_s']:.0f} s -> {a.out}/plan.json, plan.png")


if __name__ == "__main__":
    main()
