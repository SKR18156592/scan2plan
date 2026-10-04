import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from scan2plan.io import load_capture
from scan2plan.planes import detect_floor_ceiling
from scan2plan.layout import build_grids, free_space, segment_rooms, CELL
name = sys.argv[1]
cap = load_capture("../" + name); d = np.load(f"out/cloud_{name}.npz"); pts, w = d["pts"], d["w"]
f, c = detect_floor_ceiling(pts, w, np.median(cap.poses[:, 1, 3]))
cam = cap.poses[:, :3, 3]
fr, g = build_grids(pts, w, f["c"], c["c"] if c else None, np.zeros((0, 2)))
cc = fr.to_cell(fr.to_uv(cam)); g["walk"][cc[:, 1], cc[:, 0]] = True
free = free_space(g)
Ts = [1.2]
fig, ax = plt.subplots(1, len(Ts), figsize=(9, 9), squeeze=False); ax = ax[0]
for a, T in zip(ax, Ts):
    r, br = segment_rooms(free, g['tall'], max_gap=T)
    a.imshow(np.where(r > 0, r, np.nan), origin="lower", cmap="tab20", interpolation="nearest")
    a.imshow(np.where(g["tall"], 1, np.nan), origin="lower", cmap="gray"); a.imshow(np.where(br, 1, np.nan), origin="lower", cmap="autumn")
    a.set_title(f"T={T}: {r.max()} rooms " + " ".join("%.1f" % ((r == k).sum() * CELL * CELL) for k in range(1, r.max() + 1)))
fig.savefig(f"out/seg_{name}.png", dpi=70, bbox_inches="tight")
