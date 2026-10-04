"""Monocular depth vs LiDAR depth on the same frames (frame name = index)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from scan2plan.io import load_capture
cap = load_capture(sys.argv[1]); dd = Path(sys.argv[2])
rows = []
for f in sorted(dd.glob("*.npy")):
    i = int(f.stem); m = np.load(f).astype(np.float32); l = cap.depth(i); c = cap.confidence(i)
    ok = (c >= 2) & (l > 0.2) & (l < 5)
    if ok.sum() < 1000: continue
    r = m[ok] / l[ok]
    rows.append((np.median(r), np.median(np.abs(m[ok] - l[ok]) / l[ok]), np.median(np.abs(m[ok] / np.median(r) - l[ok]) / l[ok])))
R = np.array(rows)
print(f"{len(R)} frames | per-frame scale (mono/lidar) median {np.median(R[:,0]):.3f}, spread (p10..p90) {np.percentile(R[:,0],10):.3f}..{np.percentile(R[:,0],90):.3f}")
print(f"abs-rel error raw {np.median(R[:,1]):.3f} | after per-frame scale fix {np.median(R[:,2]):.3f}")
