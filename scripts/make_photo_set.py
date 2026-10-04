"""Build a photo-tier benchmark input from a LiDAR capture's video.

Real photo-tier input is 2-8 upright stills per room, one folder per room.
The sample data has no stills, so they are simulated: frames whose camera
sits inside a room of the LiDAR-tier plan are grouped into that room's
folder; up to 8 are kept, spread over viewing direction; each is rotated
upright (as EXIF orientation would do for a real photo). The LiDAR poses are
used only to pick folders, never by the photo pipeline.

    python scripts/make_photo_set.py <capture_dir> <lidar_plan.json> <out_dir>
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan2plan.io import load_capture
from scan2plan.layout import Frame2D


def main(cap_dir, plan_path, out, per_room=8, min_room_m2=2.0):
    cap = load_capture(cap_dir)
    plan = json.load(open(plan_path))
    fr = Frame2D(np.deg2rad(plan["plan_frame"]["rotation_deg"]), (0, 0), None)
    # Stray's cached poses are the drift-corrected ones the plan used.
    d = np.load(Path("out/cache") / f"{cap.root.name}_drift_on_s3.npz", allow_pickle=True)
    P = d["poses"]
    uv = fr.to_uv(P[:, :3, 3])
    yaw = np.arctan2(P[:, 0, 2], P[:, 2, 2])
    # Image "left" is up for these captures (see mono/build.orientation).
    vid = cv2.VideoCapture(str(Path(cap_dir) / "rgb.mp4"))
    frames_needed = {}
    for r in plan["rooms"]:
        if r["floor_area"]["value"] < min_room_m2:
            continue
        poly = np.array(r["polygon"], np.float32)
        inside = np.array([cv2.pointPolygonTest(poly, (float(u), float(v)), True) > 0.3 for u, v in uv])
        idx = np.flatnonzero(inside)
        if len(idx) < 2:
            continue
        # Evenly spaced in time over the room visit: the protocol asks for
        # photos from different spots that overlap their neighbours, which is
        # what consecutive positions along a walk give. (Spreading by heading
        # instead gave near-zero overlap and no baseline; SfM registered 2/34.)
        pick = idx[np.linspace(0, len(idx) - 1, min(per_room, len(idx))).round().astype(int)]
        for i in pick:
            frames_needed[int(i)] = r["id"]
    out = Path(out)
    k = 0
    want = sorted(frames_needed)
    wi = 0
    while wi < len(want):
        ok, img = vid.read()
        if not ok:
            break
        if k == want[wi]:
            room = frames_needed[k]
            dst = out / room / f"IMG_{k:06d}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), [cv2.IMWRITE_JPEG_QUALITY, 92])
            wi += 1
        k += 1
    for room in sorted(set(frames_needed.values())):
        print(room, len(list((out / room).glob("*.jpg"))), "photos")


if __name__ == "__main__":
    main(*sys.argv[1:4])
