"""2D layout from the fused cloud: wall map, dominant axes, rooms.

Everything happens in a floor-plane frame (u, v) obtained by projecting world
(x, z) and rotating so the dominant wall directions are axis-aligned.
"""
import cv2
from skimage.segmentation import watershed
import numpy as np
from scipy import ndimage as ndi

CELL = 0.05  # occupancy grid resolution, m


def dominant_angle(xz, w=None):
    """Rotation (rad) that makes walls axis-aligned: maximise histogram sharpness."""
    best = (-1, 0.0)
    for coarse in (np.deg2rad(np.arange(0, 90, 1.0)), None):
        angles = coarse if coarse is not None else best[1] + np.deg2rad(np.arange(-1, 1, 0.05))
        for a in angles:
            c, s = np.cos(a), np.sin(a)
            u = xz[:, 0] * c + xz[:, 1] * s
            v = -xz[:, 0] * s + xz[:, 1] * c
            score = 0
            for coord in (u, v):
                h, _ = np.histogram(coord, bins=np.arange(coord.min(), coord.max() + 0.02, 0.02), weights=w)
                score += (h.astype(float) ** 2).sum()
            if score > best[0]:
                best = (score, a)
    return best[1]


class Frame2D:
    """Maps world points to a rotated floor-plane grid and back."""

    def __init__(self, angle, origin, shape):
        self.angle, self.origin, self.shape = angle, np.asarray(origin), shape

    def to_uv(self, pts):
        c, s = np.cos(self.angle), np.sin(self.angle)
        x, z = pts[..., 0], pts[..., 2]
        return np.stack([x * c + z * s, -x * s + z * c], -1)

    def uv_to_world_xz(self, uv):
        c, s = np.cos(self.angle), np.sin(self.angle)
        return np.stack([uv[..., 0] * c - uv[..., 1] * s, uv[..., 0] * s + uv[..., 1] * c], -1)

    def to_cell(self, uv):
        return np.floor((uv - self.origin) / CELL).astype(int)  # (col, row)

    def cell_center(self, col, row):
        return self.origin + (np.stack([col, row], -1) + 0.5) * CELL


def build_grids(pts, w, floor_y, ceil_y, cam_uv, frame=None):
    """Occupancy grids in the (u, v) frame.

    wall: cells whose points span >= 1 m vertically inside the wall band.
          Tall vertical extent separates walls from tables/sofas/beds.
    floor: cells with points on the floor plane (observed walkable area).
    """
    h = pts[:, 1] - floor_y
    top = (ceil_y - floor_y - 0.15) if ceil_y is not None else 2.4
    if frame is None:
        band = (h > 0.3) & (h < top)
        ang = dominant_angle(pts[band][:, [0, 2]], w[band])
        tmp = Frame2D(ang, (0, 0), None)
        uv = tmp.to_uv(pts)
        lo = uv.min(0) - 0.5
        hi = uv.max(0) + 0.5
        shape = tuple(np.ceil((hi - lo) / CELL).astype(int)[::-1])  # rows, cols
        frame = Frame2D(ang, lo, shape)
    uv = frame.to_uv(pts)
    cc = frame.to_cell(uv)
    rows, cols = frame.shape
    ok = (cc[:, 0] >= 0) & (cc[:, 0] < cols) & (cc[:, 1] >= 0) & (cc[:, 1] < rows)
    idx = cc[:, 1] * cols + cc[:, 0]

    inband = ok & (h > 0.1) & (h < top)
    hmin = np.full(rows * cols, np.inf)
    hmax = np.full(rows * cols, -np.inf)
    np.minimum.at(hmin, idx[inband], h[inband])
    np.maximum.at(hmax, idx[inband], h[inband])
    span = (hmax - hmin).reshape(rows, cols)
    # Tall surfaces are walls. Shorter ones are walls only when they form a
    # long axis-aligned run (a floor-pointed scan sees only the lower wall);
    # furniture is short and compact, so it fails both tests.
    tall = span >= 1.0
    run = int(round(0.6 / CELL))
    cand = span >= 0.5
    straight = (ndi.binary_opening(cand, structure=np.ones((1, run)))
                | ndi.binary_opening(cand, structure=np.ones((run, 1))))
    # `tall` is the structural wall map used for room separation; `wall`
    # (tall or long-and-straight) is used to measure wall surfaces.
    wall = tall | straight

    on_floor = ok & (np.abs(h) < 0.04)
    floor = np.zeros(rows * cols, bool)
    floor[idx[on_floor]] = True
    floor = floor.reshape(rows, cols)

    # Anything with points in the band at all (furniture included).
    occupied = np.isfinite(span) & ~wall

    walk = np.zeros((rows, cols), bool)
    cc_cam = frame.to_cell(cam_uv)
    walk[np.clip(cc_cam[:, 1], 0, rows - 1), np.clip(cc_cam[:, 0], 0, cols - 1)] = True
    return frame, dict(wall=wall, tall=tall, floor=floor, occupied=occupied, walk=walk, span=span)


def free_space(g):
    """Interior floor area: observed floor plus furniture footprints, minus walls."""
    wall = ndi.binary_closing(g["wall"], iterations=1)
    interior = g["floor"] | g["occupied"] | ndi.binary_dilation(g["walk"], iterations=3)
    interior = ndi.binary_closing(interior, structure=np.ones((3, 3)), iterations=3)
    interior &= ~ndi.binary_dilation(wall, iterations=1)
    interior = ndi.binary_fill_holes(interior)
    interior &= ~ndi.binary_dilation(wall, iterations=1)
    lab, n = ndi.label(interior)
    if n == 0:
        return interior
    # Keep components with real observed floor (rooms seen through a doorway
    # count; specks of floor seen through windows do not).
    floor_m2 = ndi.sum(g["floor"], lab, range(1, n + 1)) * CELL * CELL
    keep = 1 + np.flatnonzero(floor_m2 >= 1.0)
    return np.isin(lab, keep)


def close_doorways(wall, max_gap=1.2):
    """Bridge gaps of up to `max_gap` metres between collinear wall pieces.

    Walls are axis-aligned after rotation, so a 1-D closing along each axis
    joins wall ends across a doorway. The cells added are the door bridges.
    """
    w = ndi.binary_dilation(wall, structure=np.ones((3, 3)))
    n = int(round(max_gap / CELL))
    run = int(round(0.3 / CELL))
    # Only a horizontal wall piece may be extended horizontally (and likewise
    # vertically), otherwise two parallel walls closer than max_gap would be
    # joined across the room between them.
    horiz = ndi.binary_opening(w, structure=np.ones((1, run)))
    vert = ndi.binary_opening(w, structure=np.ones((run, 1)))
    closed = (ndi.binary_closing(horiz, structure=np.ones((1, n)))
              | ndi.binary_closing(vert, structure=np.ones((n, 1))))
    closed |= w
    bridges = closed & ~w
    return closed, bridges


def segment_rooms(free, wall, max_gap=1.2, min_room_m2=1.0):
    """Rooms are connected components of free space once doorways are bridged."""
    closed, bridges = close_doorways(wall, max_gap)
    lab, n = ndi.label(free & ~closed)
    out = np.zeros_like(lab)
    nxt = 1
    for k in range(1, n + 1):
        if (lab == k).sum() * CELL * CELL >= min_room_m2:
            out[lab == k] = nxt
            nxt += 1
    # Give back the free cells under the dilated walls/bridges to the nearest room.
    return _fill_boundaries(out, free), bridges


def _fill_boundaries(rooms, free):
    for _ in range(20):
        hole = free & (rooms == 0)
        if not hole.any():
            break
        grown = ndi.grey_dilation(rooms, size=3)
        rooms = np.where(hole, grown, rooms)
    return rooms
