"""Horizontal plane detection: floor and ceiling.

ARKit's world frame is gravity-aligned (y up), so floor and ceiling are
horizontal planes and show up as peaks in the height histogram. The peak is
then refined with a robust plane fit so a slight tilt is measured, not assumed.
"""
import numpy as np


def _height_peaks(y, w, bin_size=0.01, min_frac=0.05):
    edges = np.arange(y.min(), y.max() + bin_size, bin_size)
    h, _ = np.histogram(y, bins=edges, weights=w)
    h = np.convolve(h, [0.25, 0.5, 0.25], mode="same")
    centers = (edges[:-1] + edges[1:]) / 2
    is_peak = (h > np.roll(h, 1)) & (h >= np.roll(h, -1)) & (h > min_frac * h.max())
    return centers[is_peak], h[is_peak]


def fit_horizontal(pts, y0, tol=0.05, iters=3):
    """Fit y = a*x + b*z + c to points near height y0; returns (c, a, b, residual std, inliers)."""
    m = np.abs(pts[:, 1] - y0) < tol
    for _ in range(iters):
        A = np.c_[pts[m, 0], pts[m, 2], np.ones(m.sum())]
        coef, *_ = np.linalg.lstsq(A, pts[m, 1], rcond=None)
        r = pts[:, 1] - (np.c_[pts[:, 0], pts[:, 2], np.ones(len(pts))] @ coef)
        sd = 1.4826 * np.median(np.abs(r[m]))
        m = np.abs(r) < max(3 * sd, 0.01)
    return dict(a=coef[0], b=coef[1], c=coef[2], std=float(sd), inliers=m)


def area_m2(pts, cell=0.1):
    """Area covered by points projected on the floor plane."""
    return len(np.unique(np.floor(pts[:, [0, 2]] / cell).astype(np.int64), axis=0)) * cell * cell


def detect_floor_ceiling(pts, w, cam_y, min_ceiling_area=2.0):
    """Floor is the lowest strong peak below the camera; ceiling the highest one above it.

    `cam_y` is the median camera height; handheld capture keeps the phone
    roughly 1.2-1.6 m above the floor, which bounds where the floor can be.
    """
    peaks, mass = _height_peaks(pts[:, 1], w)
    below = peaks[peaks < cam_y - 0.6]
    if len(below) == 0:
        raise RuntimeError("no floor plane found")
    floor = fit_horizontal(pts, below.min())
    ceiling = None
    above = peaks[peaks > cam_y + 0.2]
    for y0 in sorted(above, reverse=True):
        cand = fit_horizontal(pts, y0)
        if area_m2(pts[cand["inliers"]]) >= min_ceiling_area:
            ceiling = cand
            break
    return floor, ceiling
