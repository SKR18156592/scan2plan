"""Keyframes from a handheld video: fixed time step, sharpest frame per window."""
from pathlib import Path

import cv2
import numpy as np


def extract(video, out_dir, every_s=0.5, max_width=1440):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    win = max(int(round(fps * every_s)), 1)
    best, best_score, k, saved = None, -1, 0, []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        # Sharpness = variance of the Laplacian on a small grey copy.
        g = cv2.cvtColor(cv2.resize(img, (480, 360)), cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(g, cv2.CV_64F).var()
        if score > best_score:
            best, best_score, best_k = img, score, k
        k += 1
        if k % win == 0:
            saved.append(_save(best, best_k, out_dir, max_width))
            best, best_score = None, -1
    if best is not None:
        saved.append(_save(best, best_k, out_dir, max_width))
    return saved, fps


def _save(img, k, out_dir, max_width):
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    name = f"{k:06d}.jpg"
    cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return name
