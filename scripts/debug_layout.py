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
from scan2plan.layout import Frame2D
fr, g = build_grids(pts, w, f["c"], c["c"] if c else None, None if False else np.zeros((0, 2)))
g["walk"][:] = False
cc = fr.to_cell(fr.to_uv(cam)); g["walk"][cc[:, 1], cc[:, 0]] = True
free = free_space(g); rooms = segment_rooms(free)
print("angle deg", np.rad2deg(fr.angle), "rooms", rooms.max())
for k in range(1, rooms.max() + 1): print(" room", k, "%.2f m2" % ((rooms == k).sum() * CELL * CELL))
fig, ax = plt.subplots(1, 3, figsize=(21, 7))
ax[0].imshow(g["wall"], origin="lower"); ax[0].set_title("wall")
ax[1].imshow(free, origin="lower"); ax[1].set_title("free")
ax[2].imshow(np.where(rooms > 0, rooms, np.nan), origin="lower", cmap="tab20"); ax[2].imshow(np.where(g["wall"], 1, np.nan), origin="lower", cmap="gray")
fig.savefig(f"out/layout_{name}.png", dpi=90, bbox_inches="tight")
