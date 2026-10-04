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


def run(capture_dir, out_dir, drift_correction=True, step=3, cache_dir="out/cache", damage=True):
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
    title = f"{cap.root.name} - LiDAR tier, drift correction {'on' if drift_correction else 'off'}"
    dmg = None
    if damage:
        from dataclasses import replace
        from . import damage as D
        from .mono.build import image_up_rotation
        capP = replace(cap, poses=P)

        def dmg(result):
            # One keyframe per second; the LiDAR depth frame of the same index
            # places each detection.
            fps = len(cap) / max(cap.timestamps[-1] - cap.timestamps[0], 1e-6)
            idx = list(range(0, len(cap), max(int(round(fps)), 1)))
            paths = D.frames_from_video(cap.root / "rgb.mp4", idx, Path(out_dir) / "work" / "keyframes")
            rot = image_up_rotation(P[:, :3, :3], np.array([0.0, 1.0, 0.0]))
            return D.run(capP, result, paths, rot, Path(out_dir) / "work", errors.LIDAR, min_views=2)
    return analyze(cap.root.name, "lidar", pts, w, P, tag, timing, t0, out_dir, errors.LIDAR, title=title, damage=dmg)


def analyze(name, tier, pts, w, P, drift_tag, timing, t0, out_dir, prof, extra_scale=0.0,
            room_labels=None, meta=None, title=None, damage=None):
    """Shared by all tiers: fused cloud + camera poses -> plan JSON + render.

    room_labels(frame, grids, free) may supply the room partition (photo
    tier: one room per folder); otherwise rooms are segmented from geometry.
    extra_scale adds a tier-specific relative scale uncertainty (mono depth).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = drift_tag
    t = time.time()
    cam = P[:, :3, 3]
    floor, ceil = detect_floor_ceiling(pts, w, float(np.median(cam[:, 1])))
    frame, g = build_grids(pts, w, floor["c"], ceil["c"] if ceil else None, np.zeros((0, 2)))
    cc = frame.to_cell(frame.to_uv(cam))
    rows, cols = frame.shape
    ok = (cc[:, 0] >= 0) & (cc[:, 0] < cols) & (cc[:, 1] >= 0) & (cc[:, 1] < rows)
    g["walk"][cc[ok, 1], cc[ok, 0]] = True
    free = free_space(g)
    if room_labels is not None:
        labels = room_labels(frame, g, free)
    else:
        labels, bridges = segment_rooms(free, g["tall"])
    timing["layout_s"] = time.time() - t

    t = time.time()
    # Global wall planes, detected once per capture (see docs/FIX_DECLARATION.md).
    from . import walls as W
    hh = pts[:, 1] - floor["c"]
    top = (ceil["c"] - floor["c"] - 0.15) if ceil else 2.4
    band = (hh > 0.15) & (hh < top) & (w >= 2)
    planes = W.detect(frame.to_uv(pts[band]), hh[band], top)
    timing["planes"] = len(planes)
    rooms = []
    for k in range(1, labels.max() + 1):
        r = measure_room(k, labels == k, frame, pts, floor, ceil, planes, g["floor"], prof, extra_scale)
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
                                width=_measure(op["width"], float(np.hypot(prof.opening_edge_sigma * np.sqrt(2), np.hypot(prof.scale_sigma, extra_scale) * op["width"]))),
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
                              floor_area=_measure(r["area"], r["area_sigma"]), ceiling_height=ceiling, walls=walls,
                              floor_y_world=round(float(r["floor_y"]), 4)))

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
        capture=name,
        tier=tier,
        error_profile=dict(prof.__dict__, extra_scale_sigma=extra_scale),
        units="m",
        interval="95% (value ± 1.96 sigma)",
        plan_frame=dict(rotation_deg=round(float(np.rad2deg(frame.angle)), 3),
                        note="plan (u, v) = world (x, z) rotated so dominant walls are axis-aligned"),
        drift_correction=tag,
        floor=dict(plane_residual_std=round(floor["std"], 4)),
        quality=dict(wall_surface_spread_median=round(float(np.median([w["surface_spread"] for r in rooms for w in r["walls"]
                                                                        if w["observed"] and w["surface_spread"] is not None] or [np.nan])), 4)),
        **({"tier_details": meta} if meta else {}),
        rooms=out_rooms,
        adjacency=adj,
        property=dict(room_count=len(rooms), footprint=_measure(footprint, footprint_sigma)),
        damage_regions=[],
        concealed_damage_flags=[],
        scope_items=[],
        timing_s={k: round(v, 2) for k, v in timing.items()},
    )
    if damage is not None:
        t = time.time()
        regions, flags, items, info = damage(result)
        result.update(damage_regions=regions, concealed_damage_flags=flags, scope_items=items, damage_detection=info)
        result["timing_s"]["damage_s"] = round(time.time() - t, 2)
    result["timing_s"]["total_s"] = round(time.time() - t0, 2)
    with open(out_dir / "plan.json", "w") as f:
        json.dump(result, f, indent=2)
    render_plan(result, out_dir / "plan.png", title=title or f"{name} - {tier} tier")
    return result
