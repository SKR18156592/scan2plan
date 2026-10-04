import sys, numpy as np
sys.path.insert(0, ".")
from scan2plan.io import load_capture
from scan2plan.planes import detect_floor_ceiling
from scan2plan.layout import build_grids, free_space, segment_rooms
from scan2plan.rooms import measure_room
name = sys.argv[1]
cap = load_capture("../" + name); d = np.load(f"out/cloud_{name}.npz"); pts, w = d["pts"], d["w"]
f, c = detect_floor_ceiling(pts, w, np.median(cap.poses[:, 1, 3]))
fr, g = build_grids(pts, w, f["c"], c["c"] if c else None, np.zeros((0, 2)))
cc = fr.to_cell(fr.to_uv(cap.poses[:, :3, 3])); g["walk"][cc[:, 1], cc[:, 0]] = True
free = free_space(g); rooms, br = segment_rooms(free, g["tall"])
for k in range(1, rooms.max() + 1):
    r = measure_room(k, rooms == k, fr, pts, f, c, None, g["floor"])
    ch = r["ceiling"]
    print(f"{r['id']}: area {r['area']:.2f}±{1.96*r['area_sigma']:.2f} (mask {r['mask_area']:.2f}) ceiling", f"{ch['value']:.3f}±{1.96*ch['sigma']:.3f}" if ch else None)
    for wl in r["walls"]:
        print(f"   {wl['id']} {wl['length']:.3f}±{1.96*wl['length_sigma']:.3f} obs={wl['observed']}", [(o['type'], round(o['width'], 3)) for o in wl["openings"]])
