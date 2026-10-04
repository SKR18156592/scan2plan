"""Put 2D damage detections onto plan surfaces with metric extent, flag
concealed damage, and write scope line items keyed to surfaces.

A detection box is back-projected through the frame's depth and pose. Its
points decide the surface (floor, ceiling, or the nearest wall of the room
they fall in) and are projected onto that surface; the extent is the 5-95th
percentile span along the surface axes. Detections of the same class on the
same surface within 0.4 m are merged across frames. On LiDAR and video tiers
a region must be seen in >= 2 frames (multi-view consistency is the main
false-positive filter for a zero-shot detector); a photo may be the only view.
"""
import numpy as np

from ..layout import Frame2D

MERGE_DIST = 0.4


def _segment_dist(p, a, b):
    d = b - a
    t = np.clip(np.dot(p - a, d) / max(np.dot(d, d), 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t * d)), t


def locate(detections, cap, plan, min_views=2):
    frame = Frame2D(np.deg2rad(plan["plan_frame"]["rotation_deg"]), (0, 0), None)
    rooms = plan["rooms"]
    regions = []
    for det in detections:
        i = det["image"]
        d = cap.depth(i)
        H, W = d.shape
        x0, y0, x1, y1 = det["box"]
        # Central 60% of the box: box edges include background.
        cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 0.3, (y1 - y0) * 0.3
        u0, u1 = int((cx - hw) * W), int(np.ceil((cx + hw) * W))
        v0, v1 = int((cy - hh) * H), int(np.ceil((cy + hh) * H))
        if u1 <= u0 or v1 <= v0:
            continue
        K = cap.K_depth(d.shape, i)
        vv, uu = np.mgrid[v0:v1, u0:u1]
        z = d[v0:v1, u0:u1]
        ok = (z > 0.2) & (z < 5)
        if ok.sum() < 4:
            continue
        z, uu, vv = z[ok], uu[ok], vv[ok]
        # Keep the dominant depth layer (the damaged surface, not what is in front).
        zm = np.median(z)
        keep = np.abs(z - zm) < 0.15 * zm
        z, uu, vv = z[keep], uu[keep], vv[keep]
        pc = np.stack([(uu - K[0, 2]) * z / K[0, 0], (vv - K[1, 2]) * z / K[1, 1], z], 1)
        T = cap.poses[i]
        pw = pc @ T[:3, :3].T + T[:3, 3]
        # Full box extent on the surface: scale the central 60% back up.
        region = _assign(pw, frame, rooms)
        if region is None:
            continue
        region.update(cls=det["cls"], score=det["score"], frames=[i])
        region["extent"] = [e / 0.6 for e in region["extent"]]
        regions.append(region)
    merged = _merge(regions)
    return [r for r in merged if len(set(r["frames"])) >= min_views]


def _assign(pw, frame, rooms):
    uv = frame.to_uv(pw)
    c = uv.mean(0)
    # Room containing the centroid (or nearest room).
    best = None
    for r in rooms:
        P = np.array(r["polygon"])
        inside = _point_in_poly(c, P)
        dmin = min(_segment_dist(c, np.array(w["start"]), np.array(w["end"]))[0] for w in r["walls"])
        key = (0 if inside else 1, dmin)
        if best is None or key < best[0]:
            best = (key, r)
    room = best[1]
    floor_y = room.get("floor_y_world")
    ceil_h = room["ceiling_height"]["value"]
    h = pw[:, 1] - floor_y if floor_y is not None else None
    if h is not None:
        hm = np.median(h)
        if hm < 0.08:
            ext = np.percentile(uv, 95, axis=0) - np.percentile(uv, 5, axis=0)
            return dict(room=room["id"], surface=f"{room['id']}_floor", surface_type="floor", centroid=c.tolist(),
                        height=float(hm), extent=[float(ext[0]), float(ext[1])])
        if (ceil_h is not None and hm > ceil_h - 0.12) or (ceil_h is None and hm > 2.2):
            ext = np.percentile(uv, 95, axis=0) - np.percentile(uv, 5, axis=0)
            return dict(room=room["id"], surface=f"{room['id']}_ceiling", surface_type="ceiling", centroid=c.tolist(),
                        height=float(hm), extent=[float(ext[0]), float(ext[1])])
    # Nearest wall of that room.
    wbest = None
    for w in room["walls"]:
        a, b = np.array(w["start"]), np.array(w["end"])
        dist, t = _segment_dist(c, a, b)
        if wbest is None or dist < wbest[0]:
            wbest = (dist, w, a, b)
    dist, w, a, b = wbest
    if dist > 0.5:
        return None  # not on a surface of the plan (furniture, clutter)
    d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    along = (uv - a) @ d
    span = np.percentile(along, 95) - np.percentile(along, 5)
    vert = (np.percentile(h, 95) - np.percentile(h, 5)) if h is not None else span
    return dict(room=room["id"], surface=w["id"], surface_type="wall", centroid=c.tolist(),
                height=float(np.median(h)) if h is not None else None,
                along=[float(np.percentile(along, 5)), float(np.percentile(along, 95))],
                extent=[float(span), float(vert)], wall_length=w["length"]["value"],
                near_opening=_near_opening_corner(w, along, h))


def _near_opening_corner(w, along, h):
    """True if the region touches the corner of an opening on its wall
    (cracks running from door/window corners are the structural pattern)."""
    if h is None:
        return False
    for op in w["openings"]:
        lo, hi = op["offset"], op["offset"] + op["width"]["value"]
        for corner in (lo, hi):
            if np.min(np.abs(along - corner)) < 0.3:
                return True
    return False


def _point_in_poly(p, P):
    x, y = p
    inside = False
    n = len(P)
    for k in range(n):
        x1, y1 = P[k]
        x2, y2 = P[(k + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _merge(regions):
    out = []
    for r in sorted(regions, key=lambda r: -r["score"]):
        for m in out:
            if (m["cls"] == r["cls"] and m["surface"] == r["surface"]
                    and np.linalg.norm(np.array(m["centroid"]) - np.array(r["centroid"])) < MERGE_DIST):
                m["frames"] += r["frames"]
                m["score"] = max(m["score"], r["score"])
                m["extent"] = [max(a, b) for a, b in zip(m["extent"], r["extent"])]
                break
        else:
            out.append(dict(r))
    return out
