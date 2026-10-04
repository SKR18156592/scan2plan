import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from scan2plan.io import load_capture
from scan2plan.planes import detect_floor_ceiling
from scan2plan.layout import dominant_angle, Frame2D
name = sys.argv[1]
cap = load_capture("../" + name); d = np.load(f"out/cloud_{name}.npz"); pts, w = d["pts"], d["w"]
f, c = detect_floor_ceiling(pts, w, np.median(cap.poses[:, 1, 3]))
h = pts[:, 1] - f["c"]; band = (h > 0.3) & (h < 2.2)
ang = dominant_angle(pts[band][:, [0, 2]], w[band]); fr = Frame2D(ang, (0, 0), None)
uv = fr.to_uv(pts[band]); cam = fr.to_uv(cap.poses[:, :3, 3])
fig, ax = plt.subplots(figsize=(14, 14))
ax.scatter(uv[:, 0], uv[:, 1], s=0.02, c=h[band], cmap="viridis")
ax.plot(cam[:, 0], cam[:, 1], "r-", lw=0.6)
for k in range(0, len(cam), 500): ax.annotate(str(k), cam[k], color="r", fontsize=8)
ax.set_aspect("equal"); ax.grid(True, lw=0.3); ax.set_xticks(np.arange(np.floor(uv[:,0].min()), uv[:,0].max(), 0.5)); ax.set_yticks(np.arange(np.floor(uv[:,1].min()), uv[:,1].max(), 0.5))
ax.tick_params(labelsize=6)
fig.savefig(f"out/walls_{name}.png", dpi=80, bbox_inches="tight")
