"""Per-fragment wall direction over time: yaw drift shows as a trend."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from scan2plan.io import load_capture
from scan2plan import drift
from scan2plan.layout import dominant_angle

def frag_angles(cap, poses, frags):
    out = []
    for f in frags:
        if f is None: out.append(np.nan); continue
        idx = range(f["start"], f["end"], 5)
        from dataclasses import replace
        c2 = replace(cap, poses=poses)
        from scan2plan.io import backproject
        P = np.concatenate([backproject(c2, i, max_depth=3.5) for i in idx])
        wp, wn = drift._wall_points(P[::3])
        if len(wp) < 500: out.append(np.nan); continue
        a = np.rad2deg(dominant_angle(wp[:, [0, 2]]))
        out.append(a)
    return np.array(out)

name = sys.argv[1]
cap = load_capture("../" + name)
frags = drift.fragments(cap)
d = np.load(f"out/cache/{name}_drift_on_s3.npz", allow_pickle=True)
raw = frag_angles(cap, cap.poses, frags)
cor = frag_angles(cap, d["poses"], frags)
ref = np.nanmedian(raw)
wrap = lambda a: (a - ref + 45) % 90 - 45
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(wrap(raw), ".-", label="raw ARKit"); ax.plot(wrap(cor), ".-", label="loop-closure corrected")
ax.set_xlabel("fragment (~2 s)"); ax.set_ylabel("wall direction - median (deg)"); ax.legend(); ax.grid(alpha=.3)
fig.savefig(f"out/yaw_{name}.png", dpi=80, bbox_inches="tight")
for lab, a in (("raw", raw), ("cor", cor)):
    w = wrap(a); w = w[np.isfinite(w)]
    print(name, lab, "std %.2f deg, p5..p95 %.2f..%.2f" % (w.std(), *np.percentile(w, [5, 95])))
