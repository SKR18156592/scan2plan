"""Quick look: accumulate a subsampled point cloud and render a top-down view."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan2plan.io import backproject, load_capture

cap = load_capture(sys.argv[1])
step = int(sys.argv[2]) if len(sys.argv) > 2 else 10
pts = np.concatenate([backproject(cap, i)[::4] for i in range(0, len(cap), step)])
y = pts[:, 1]
print(f"{len(pts)} pts, height (y) percentiles 1/50/99:", np.percentile(y, [1, 50, 99]).round(3))
hist, edges = np.histogram(y, bins=np.arange(y.min(), y.max() + 0.02, 0.02))
for k in np.argsort(hist)[-6:][::-1]:
    print(f"  height bin {edges[k]:+.2f} m: {hist[k]} pts")

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].scatter(pts[:, 0], pts[:, 2], s=0.05, c=y, cmap="viridis")
ax[0].plot(cap.poses[:, 0, 3], cap.poses[:, 2, 3], "r-", lw=0.8)
ax[0].set_aspect("equal"); ax[0].set_title("all points, top-down (x,z), camera path red")
lo, hi = np.percentile(y, [1, 99])
band = (y > lo + 0.3) & (y < hi - 0.3)
ax[1].scatter(pts[band, 0], pts[band, 2], s=0.05, c="k")
ax[1].set_aspect("equal"); ax[1].set_title("wall band (excludes floor/ceiling)")
out = Path("out") / f"topdown_{cap.root.name}.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight")
print("wrote", out)
