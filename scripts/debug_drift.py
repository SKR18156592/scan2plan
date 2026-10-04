import sys, time, json, numpy as np
sys.path.insert(0, ".")
from scan2plan.io import load_capture
from scan2plan import drift
name = sys.argv[1]
cap = load_capture("../" + name)
t = time.time(); log = []
poses, s = drift.correct(cap, log=log)
print(json.dumps(s, indent=1), f"{time.time()-t:.0f}s")
acc = [l for l in log if l["accepted"]]
print(f"{len(log)} candidate pairs, {len(acc)} accepted")
for l in sorted(acc, key=lambda l: -abs(l["yaw_deg"]))[:8]:
    print(f"  {l["a"]:3d}->{l["b"]:3d} cond {l["cond"]:.2f} fit {l["fitness"]:.2f} rmse {l['rmse']*100:.1f}cm t {np.round(l['t'],3)} yaw {l['yaw_deg']:.2f}")
np.save(f"out/poses_corrected_{name}.npy", poses)
