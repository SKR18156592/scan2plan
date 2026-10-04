"""Global wall planes in the axis-aligned plan frame.

A wall face is a vertical plane u = c (or v = c). It is detected once per
capture from all wall-band points, so its position does not depend on which
room is being measured or how much floor the capture happened to see.
Room outlines are then snapped onto these planes (rooms.py).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class WallPlane:
    axis: int  # 0: plane u = coord (a "v" edge in rooms.py), 1: plane v = coord ("h" edge)
    coord: float
    segments: list  # [(lo, hi)] extents along the other axis
    n: int
    std: float

    def overlap(self, lo, hi):
        return sum(max(0.0, min(hi, b) - max(lo, a)) for a, b in self.segments)


def detect(uv, h, top, min_len=0.3, min_cover=0.5, bin_size=0.01, merge_gap=0.25):
    """Find wall faces from wall-band points (uv: (N, 2) plan coords, h: height above floor)."""
    planes = []
    for axis in (0, 1):
        x, y = uv[:, axis], uv[:, 1 - axis]
        hist, edges = np.histogram(x, bins=np.arange(x.min(), x.max() + bin_size, bin_size))
        sm = np.convolve(hist, [1, 2, 3, 2, 1], mode="same") / 9
        peaks = np.flatnonzero((sm > np.roll(sm, 1)) & (sm >= np.roll(sm, -1)) & (sm > 20))
        for k in peaks:
            c0 = (edges[k] + edges[k + 1]) / 2
            near = np.abs(x - c0) < 0.03
            if near.sum() < 50:
                continue
            c = float(np.median(x[near]))
            near = np.abs(x - c) < 0.025
            # Extents along the wall: 5 cm bins with points over >= min_cover of height.
            ys, hs = y[near], h[near]
            step = 0.05
            yb = np.floor((ys - ys.min()) / step).astype(int)
            hb = np.clip((hs / 0.1).astype(int), 0, int(top / 0.1))
            occ = np.zeros((yb.max() + 1, int(top / 0.1) + 1), bool)
            occ[yb, hb] = True
            good = occ.sum(1) * 0.1 >= min_cover
            segs = _runs(good, int(merge_gap / step))
            segs = [(ys.min() + s * step, ys.min() + e * step) for s, e in segs if (e - s) * step >= min_len]
            if not segs:
                continue
            r = x[near] - c
            planes.append(WallPlane(axis, c, segs, int(near.sum()), float(1.4826 * np.median(np.abs(r)))))
    return _dedupe(planes)


def _runs(mask, max_gap):
    """Runs of True allowing gaps up to max_gap bins."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    out, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap + 1:
            out.append((s, prev + 1))
            s = i
        prev = i
    out.append((s, prev + 1))
    return out


def _dedupe(planes, tol=0.02):
    """Histogram peaks 1-2 cm apart on one face are the same plane: keep the stronger."""
    planes = sorted(planes, key=lambda p: -p.n)
    kept = []
    for p in planes:
        clash = [q for q in kept if q.axis == p.axis and abs(q.coord - p.coord) < tol
                 and any(q.overlap(a, b) > 0 for a, b in p.segments)]
        if not clash:
            kept.append(p)
    return kept


def snap(orient, coord, lo, hi, inward, planes, search_out=1.0, search_in=0.15, min_overlap=0.5):
    """Nearest wall plane on the outer side of a room edge.

    orient 'h' is an edge at v = coord, 'v' at u = coord; [lo, hi] its span;
    `inward` = +1 if the room interior is at larger coordinate. Picks the
    first plane met walking outward from just inside the edge, among planes
    that cover at least `min_overlap` of the edge's span.
    """
    axis = 1 if orient == "h" else 0
    span = max(hi - lo, 1e-6)
    best = None
    for p in planes:
        if p.axis != axis:
            continue
        out = (coord - p.coord) * inward  # positive = outside the room
        if out < -search_in or out > search_out:
            continue
        if p.overlap(lo, hi) < min_overlap * span:
            continue
        if best is None or out < best[0]:
            best = (out, p)
    return best[1] if best else None
