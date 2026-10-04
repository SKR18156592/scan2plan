"""Overlay two registered plans (repeatability evidence)."""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "scripts")
from repeatability import register
pa, pb = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
R, t, err = register(pa, pb)
fig, ax = plt.subplots(figsize=(12, 12))
for p, col, tf in ((pa, "tab:blue", lambda P: np.asarray(P)), (pb, "tab:red", lambda P: np.asarray(P) @ R.T + t)):
    for r in p["rooms"]:
        P = tf(r["polygon"]); P = np.vstack([P, P[:1]])
        ax.plot(P[:, 0], P[:, 1], color=col, lw=1.5)
        ax.annotate(f'{r["id"]} {r["floor_area"]["value"]:.1f}', P[:-1].mean(0), color=col)
ax.set_aspect("equal"); ax.set_title(f"blue: {pa['capture']}  red: {pb['capture']} (registered, median residual {100*err:.1f} cm)")
fig.savefig(sys.argv[3], dpi=70, bbox_inches="tight")
