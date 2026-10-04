"""LiDAR-tier pipeline: capture folder in, plan JSON + rendered plan out."""
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from . import errors
from .cloud import fuse
from .io import load_capture
from .layout import CELL, build_grids, free_space, segment_rooms
from .planes import detect_floor_ceiling
from .render import render_plan
from .rooms import measure_room

SCHEMA_VERSION = "0.1"


def _measure(value, sigma, **extra):
    if value is None:
        return dict(value=None, sigma=None, ci95=None, **extra)
    return dict(value=round(float(value), 4), sigma=round(float(sigma), 4),
                ci95=[round(x, 4) for x in errors.interval(value, sigma)], **extra)


def _room_at(rooms, frame, uv):
    cc = frame.to_cell(np.asarray(uv))
    r, c = cc[1], cc[0]
    if 0 <= r < rooms.shape[0] and 0 <= c < rooms.shape[1]:
        win = rooms[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3]
        vals = win[win > 0]
        if len(vals):
            return int(np.bincount(vals).argmax())
    return 0


def run(capture_dir, out_dir, drift_correction=True, step=3, cache_dir="out/cache"):
    """Process one capture. Caches (keyed by capture name and drift mode) make
    re-runs fast; delete the cache dir or pass cache_dir=None for a cold run."""
    t0 = time.time()
    timing = {}
    cap = load_capture(capture_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "drift_on" if drift_correction else "drift_off"
    cache = Path(cache_dir) / f"{cap.root.name}_{mode}_s{step}.npz" if cache_dir else None

    if cache and cache.exists():
        d = np.load(cache, allow_pickle=True)
        pts, w, P = d["pts"], d["w"], d["poses"]
        tag = d["tag"].item()
        timing["cache_hit"] = 1.0
    else:
        P, tag = cap.poses, dict(enabled=False, note="ablation: raw ARKit poses")
        if drift_correction:
            from .drift import correct
            P, tag = correct(cap)
            tag["enabled"] = True
            timing["drift_s"] = time.time() - t0
        pts, w = fuse(cap, poses=P, step=step)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, pts=pts, w=w, poses=P, tag=np.array(tag, dtype=object))
    timing["fuse_s"] = time.time() - t0

    t = time.time()
    cam = P[:, :3, 3]
    floor, ceil = detect_floor_ceiling(pts, w, float(np.median(cam[:, 1])))
    frame, g = build_grids(pts, w, floor["c"], ceil["c"] if ceil else None, np.zeros((0, 2)))
    cc = frame.to_cell(frame.to_uv(cam))
    rows, cols = frame.shape
    ok = (cc[:, 0] >= 0) & (cc[:, 0] < cols) & (cc[:, 1] >= 0) & (cc[:, 1] < rows)
    g["walk"][cc[ok, 1], cc[ok, 0]] = True
    free = free_space(g)
    labels, bridges = segment_rooms(free, g["tall"])
    timing["layout_s"] = time.time() - t

    t = time.time()
    rooms = []
    for k in range(1, labels.max() + 1):
        r = measure_room(k, labels == k, frame, pts, floor, ceil, None, g["floor"])
        rooms.append(r)
    timing["measure_s"] = time.time() - t

    # Assemble output contract.
    out_rooms, adjacency = [], {}
    for i, r in enumerate(rooms, 1):
        walls = []
        for wl in r["walls"]:
            a, b = np.array(wl["start"]), np.array(wl["end"])
            d = (b - a) / max(wl["length"], 1e-9)
            along = 0 if abs(d[0]) > abs(d[1]) else 1
            ops = []
            for j, op in enumerate(wl["openings"]):
                lo = min(a[along], b[along])
                # Offset measured from wall start along its direction.
                s = op["start"] if d[along] > 0 else op["end"]
                offset = abs(s - a[along])
                mid = a + d * (offset + op["width"] / 2)
                n = np.array([-d[1], d[0]])
                # Probe both sides; the side that is not this room is the neighbour.
                other = 0
                for sgn in (1, -1):
                    q = _room_at(labels, frame, mid + sgn * n * 0.4)
                    if q and q != i:
                        other = q
                oid = f"{wl['id']}_o{j}"
                ops.append(dict(id=oid, type=op["type"], offset=round(float(offset), 4),
                                width=_measure(op["width"], errors.OPENING_EDGE_SIGMA),
                                connects_to=f"r{other}" if other else None))
                if other:
                    adjacency.setdefault(tuple(sorted((i, other))), []).append(oid)
            walls.append(dict(id=wl["id"], start=[round(x, 4) for x in wl["start"]], end=[round(x, 4) for x in wl["end"]],
                              length=_measure(wl["length"], wl["length_sigma"]), observed=wl["observed"], openings=ops))
        ch = r["ceiling"]
        if ch:
            ceiling = _measure(ch["value"], ch["sigma"], status="measured")
        else:
            ceiling = _measure(None, None, status="not_observed",
                               note="no ceiling points in this room; capture did not sweep the ceiling")
        out_rooms.append(dict(id=r["id"], name=f"Room {i}", polygon=[[round(x, 4) for x in p] for p in r["polygon_uv"]],
                              floor_area=_measure(r["area"], r["area_sigma"]), ceiling_height=ceiling, walls=walls))

    # Rooms that touch across a bridged doorway or a thin wall are adjacent
    # even when no opening was measured on that wall.
    grown = {k: ndi.binary_dilation(labels == k, iterations=int(0.35 / CELL)) for k in range(1, labels.max() + 1)}
    for a in grown:
        for b in grown:
            if a < b and (grown[a] & (labels == b)).any():
                key = (a, b)
                if key not in adjacency:
                    adjacency[key] = []
    adj = [dict(rooms=[f"r{a}", f"r{b}"], via=v if v else "shared_wall") for (a, b), v in sorted(adjacency.items())]

    footprint = sum(r["area"] for r in rooms)
    footprint_sigma = float(np.sqrt(sum(r["area_sigma"] ** 2 for r in rooms)))
    result = dict(
        schema_version=SCHEMA_VERSION,
        capture=Path(capture_dir).name,
        tier="lidar",
        units="m",
        interval="95% (value ± 1.96 sigma)",
        plan_frame=dict(rotation_deg=round(float(np.rad2deg(frame.angle)), 3),
                        note="plan (u, v) = world (x, z) rotated so dominant walls are axis-aligned"),
        drift_correction=tag,
        floor=dict(plane_residual_std=round(floor["std"], 4)),
        quality=dict(wall_surface_spread_median=round(float(np.median([w["surface_spread"] for r in rooms for w in r["walls"] if w["observed"]])), 4)),
        rooms=out_rooms,
        adjacency=adj,
        property=dict(room_count=len(rooms), footprint=_measure(footprint, footprint_sigma)),
        damage_regions=[],
        concealed_damage_flags=[],
        scope_items=[],
        timing_s={k: round(v, 2) for k, v in timing.items()},
    )
    result["timing_s"]["total_s"] = round(time.time() - t0, 2)
    with open(out_dir / "plan.json", "w") as f:
        json.dump(result, f, indent=2)
    render_plan(result, out_dir / "plan.png", title=f"{result['capture']} - LiDAR tier, drift correction {'on' if drift_correction else 'off'}")
    return result
