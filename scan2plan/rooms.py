"""Per-room geometry: rectilinear outline, wall surfaces, openings, heights.

The room masks from layout.py are only good to a grid cell (5 cm). Every
dimension reported here is re-measured from the raw points instead: each wall
edge is moved onto the wall surface found in the points, and openings are gaps
in that surface.
"""
import cv2
import numpy as np

from . import errors
from .layout import CELL

Z95 = 1.96


def outline(mask, frame, eps=0.15):
    """Rectilinear polygon (list of (u, v)) approximating a room mask."""
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    poly = cv2.approxPolyDP(c, eps / CELL, True)[:, 0, :].astype(float)
    # Cell indices to (u, v) at cell corners.
    uv = frame.origin + (poly + 0.5) * CELL
    # Snap each edge to horizontal (v = const) or vertical (u = const).
    edges = []
    n = len(uv)
    for i in range(n):
        a, b = uv[i], uv[(i + 1) % n]
        d = b - a
        if abs(d[0]) >= abs(d[1]):
            edges.append(["h", (a[1] + b[1]) / 2, abs(d[0])])
        else:
            edges.append(["v", (a[0] + b[0]) / 2, abs(d[1])])
    # Merge consecutive same-orientation edges (length-weighted coordinate).
    merged = True
    while merged and len(edges) > 4:
        merged = False
        for i in range(len(edges)):
            j = (i + 1) % len(edges)
            if edges[i][0] == edges[j][0]:
                L = edges[i][2] + edges[j][2]
                edges[i] = [edges[i][0], (edges[i][1] * edges[i][2] + edges[j][1] * edges[j][2]) / max(L, 1e-9), L]
                del edges[j]
                merged = True
                break
    return simplify([(o, c) for o, c, _ in edges])


def simplify(edges, min_len=0.3):
    """Remove jogs: drop the shortest edge under `min_len` and merge its two
    neighbours (same orientation, so they become one wall), until none left.

    Clutter along a wall makes the free-space mask ragged; real rooms rarely
    have wall offsets under 30 cm (a column or chase is the exception and
    is accepted as a known failure mode).
    """
    edges = list(edges)
    while len(edges) > 4:
        P = vertices(edges)
        n = len(edges)
        L = np.array([np.linalg.norm(P[(k + 1) % n] - P[k]) for k in range(n)])
        k = int(np.argmin(L))
        if L[k] >= min_len:
            break
        i, j = (k - 1) % n, (k + 1) % n
        # Neighbours i and j are parallel; merge them weighted by length.
        wi, wj = L[i], L[j]
        c = (edges[i][1] * wi + edges[j][1] * wj) / max(wi + wj, 1e-9)
        new = (edges[i][0], c)
        keep = [e for t, e in enumerate(edges) if t not in (i, k, j)]
        # Re-insert merged edge at position of i (ordering preserved).
        order = [t for t in range(n) if t not in (k, j)]
        edges = [new if t == i else edges[t] for t in order]
    return edges


def vertices(edges):
    """Vertex k is where edge k-1 meets edge k (alternating h/v)."""
    out = []
    for k in range(len(edges)):
        (o1, c1), (o2, c2) = edges[k - 1], edges[k]
        out.append((c2, c1) if o1 == "h" else (c1, c2))
    return np.array(out)


def polygon_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _inside_sign(P, k, o):
    """+1 if the room interior lies on the increasing-coordinate side of edge k."""
    a, b = P[k], P[(k + 1) % len(P)]
    mid = (a + b) / 2
    centroid = P.mean(0)
    axis = 1 if o == "h" else 0
    # Use a point-in-polygon probe rather than the centroid (L-shapes).
    probe = mid.copy()
    probe[axis] += 0.05
    inside = cv2.pointPolygonTest(P.astype(np.float32), tuple(map(float, probe)), False) >= 0
    return 1 if inside else -1


def refine_edges(edges, wall_uv, wall_h, search=0.35):
    """Move each edge onto the nearest wall surface found in the points.

    For edge k, take wall-band points over the middle of its span and within
    `search` metres outward. The wall surface is the first dense layer met
    when walking outward from the room (the room-facing side of the wall).
    """
    P = vertices(edges)
    out = []
    for k, (o, c) in enumerate(edges):
        a, b = P[k], P[(k + 1) % len(P)]
        along = 0 if o == "h" else 1
        perp = 1 - along
        lo, hi = sorted((a[along], b[along]))
        shrink = min(0.15, (hi - lo) / 4)
        sgn = _inside_sign(P, k, o)  # interior direction along `perp`
        rel = (wall_uv[:, perp] - c) * -sgn  # positive = outward
        sel = ((wall_uv[:, along] > lo + shrink) & (wall_uv[:, along] < hi - shrink)
               & (rel > -0.15) & (rel < search))
        info = dict(orient=o, coord=c, observed=False, n=int(sel.sum()), std=None, inward=sgn)
        hist = None
        if sel.sum() >= 30:
            hist, edges_ = np.histogram(rel[sel], bins=np.arange(-0.15, search + 0.01, 0.01))
        if hist is not None and hist.max() >= 10:
            r = rel[sel]
            thr = max(10, 0.25 * hist.max())
            first = np.flatnonzero(hist >= thr)[0]
            peak = (edges_[first] + edges_[first + 1]) / 2
            layer = np.abs(r - peak) < 0.03
            surf = np.median(r[layer])
            sd = 1.4826 * np.median(np.abs(r[layer] - surf))
            info.update(coord=c - sgn * surf, observed=True, n=int(layer.sum()), std=float(sd))
        out.append(info)
    return out


def wall_sigma(info):
    """1-sigma uncertainty (m) of a wall surface position."""
    if not info["observed"]:
        return errors.UNOBSERVED_WALL_SIGMA
    # Statistical error of the median plus the per-surface LiDAR bias floor.
    stat = 1.2533 * info["std"] / np.sqrt(max(info["n"], 1))
    return float(np.hypot(stat, errors.LIDAR_SURFACE_BIAS))


def find_openings(P, k, info, pts_uv, pts_h, floor_obs, frame, ceiling_h):
    """Gaps in wall k's surface that run from the floor up past door height.

    Samples the wall line every 2.5 cm and checks for surface points in a low
    (0.1-0.7 m) and mid (0.9-1.8 m) band. Door/opening: both bands empty and
    the floor on the far side is observed (otherwise it is just unscanned).
    Window: low band present, mid band empty, on a wall whose mid band is
    otherwise well covered.
    """
    o, c = info["orient"], info["coord"]
    along = 0 if o == "h" else 1
    perp = 1 - along
    a, b = P[k], P[(k + 1) % len(P)]
    lo, hi = sorted((a[along], b[along]))
    near = np.abs(pts_uv[:, perp] - c) < 0.06
    step = 0.025
    bins = np.arange(lo, hi + step, step)
    if len(bins) < 4:
        return []

    def cover(hmin, hmax):
        m = near & (pts_h > hmin) & (pts_h < hmax)
        h, _ = np.histogram(pts_uv[m, along], bins=bins)
        return h > 0

    low, mid = cover(0.1, 0.7), cover(0.9, min(1.8, (ceiling_h or 2.4) - 0.2))
    centers = (bins[:-1] + bins[1:]) / 2
    out = []
    # Far-side floor check uses the grid.
    sgn = info["inward"]
    rows, cols = frame.shape

    def far_floor(s):
        q = np.zeros(2)
        q[along] = s
        q[perp] = c - sgn * 0.3
        cc = frame.to_cell(q)
        r0, c0 = cc[1], cc[0]
        win = floor_obs[max(r0 - 3, 0):r0 + 4, max(c0 - 3, 0):c0 + 4]
        return win.any()

    def runs(mask):
        m = np.r_[False, mask, False].astype(int)
        d = np.diff(m)
        return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))

    gap = ~low & ~mid
    for s, e in runs(gap):
        # Ignore gaps touching the corners: corners are often occluded.
        if s == 0 or e == len(gap):
            continue
        width = (e - s) * step
        if width < 0.55 or width > 3.0:
            continue
        if not far_floor(centers[(s + e) // 2]):
            continue
        out.append(dict(type="door" if width <= 1.2 else "opening", start=centers[s] - step / 2,
                        end=centers[e - 1] + step / 2, width=width))
    if mid.mean() > 0.6:
        for s, e in runs(low & ~mid):
            width = (e - s) * step
            if 0.4 <= width <= 3.0 and s > 0 and e < len(mid):
                out.append(dict(type="window", start=centers[s] - step / 2, end=centers[e - 1] + step / 2, width=width))
    return out


def measure_room(rid, mask, frame, pts, floor_fit, ceil_fit, wall_mask_uv, floor_obs):
    """Full measurement for one room. pts are world points (N, 3)."""
    uv_all = frame.to_uv(pts)
    cc = frame.to_cell(uv_all)
    rows, cols = frame.shape
    ok = (cc[:, 0] >= 0) & (cc[:, 0] < cols) & (cc[:, 1] >= 0) & (cc[:, 1] < rows)
    in_room = np.zeros(len(pts), bool)
    in_room[ok] = mask[cc[ok, 1], cc[ok, 0]]

    # Per-room floor height: the floor plane can step between rooms
    # (thresholds, bathrooms), so it is measured inside the room footprint.
    y = pts[:, 1]
    fl = in_room & (np.abs(y - floor_fit["c"]) < 0.06)
    floor_y = float(np.median(y[fl])) if fl.sum() > 50 else floor_fit["c"]
    h = y - floor_y

    ceiling = None
    if ceil_fit is not None:
        ce = in_room & (np.abs(y - ceil_fit["c"]) < 0.08)
        if ce.sum() > 200:
            cy = np.median(y[ce])
            sd_f = 1.4826 * np.median(np.abs(y[fl] - floor_y)) if fl.sum() > 50 else 0.02
            sd_c = 1.4826 * np.median(np.abs(y[ce] - cy))
            sigma = errors.ceiling_sigma(cy - floor_y, sd_f, fl.sum(), sd_c, ce.sum())
            ceiling = dict(value=float(cy - floor_y), sigma=sigma, n_floor=int(fl.sum()), n_ceiling=int(ce.sum()))

    # Wall-band points near this room (dilated footprint) drive edge refinement.
    near = np.zeros(len(pts), bool)
    from scipy import ndimage as ndi
    dil = ndi.binary_dilation(mask, iterations=int(0.5 / CELL))
    near[ok] = dil[cc[ok, 1], cc[ok, 0]]
    top = (ceiling["value"] - 0.15) if ceiling else 2.4
    band = near & (h > 0.15) & (h < top)
    edges = outline(mask, frame)
    infos = refine_edges(edges, uv_all[band], h[band])
    P = vertices([(i["orient"], i["coord"]) for i in infos])
    sig = [wall_sigma(i) for i in infos]

    walls = []
    n = len(infos)
    area = polygon_area(P)
    var_area = 0.0
    for k, info in enumerate(infos):
        a, b = P[k], P[(k + 1) % n]
        L = float(np.linalg.norm(b - a))
        # Length is set by the two neighbouring (perpendicular) walls.
        s_len = errors.length_sigma(L, sig[k - 1], sig[(k + 1) % n])
        var_area += (L * sig[k]) ** 2
        ops = find_openings(P, k, info, uv_all[near], h[near], floor_obs, frame, ceiling["value"] if ceiling else None)
        walls.append(dict(id=f"r{rid}_w{k}", start=a.tolist(), end=b.tolist(), length=L, length_sigma=s_len,
                          observed=info["observed"], surface_sigma=sig[k], surface_spread=info["std"], openings=ops))
    s_area = float(np.sqrt(var_area + (errors.SCALE_SIGMA * 2 * area) ** 2))
    return dict(id=f"r{rid}", polygon_uv=P.tolist(), floor_y=floor_y, area=float(area), area_sigma=s_area,
                ceiling=ceiling, walls=walls, mask_area=float(mask.sum() * CELL * CELL))
