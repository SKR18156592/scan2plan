"""Error model. Every reported measurement gets a 1-sigma value from here and
is published as a 95% interval (value +- 1.96 sigma).

One Profile per input tier. The LiDAR constants are priors from published
iPhone LiDAR/ARKit evaluations; the photo/video constants are set from the
measured mono-vs-LiDAR depth error on the sample captures (see
scripts/eval_mono_depth.py and the calibration section of the report).
None of them are fitted to tape ground truth yet: there is none for the
sample data. They are the numbers to recalibrate once it exists.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Profile:
    name: str
    # Per-surface position bias after averaging all observations of it.
    surface_bias: float
    # Relative scale error applied to every length.
    scale_sigma: float
    # Edge with no supporting points: keeps the mask boundary.
    unobserved_wall_sigma: float
    # Opening edge localisation (one edge).
    opening_edge_sigma: float


LIDAR = Profile("lidar", surface_bias=0.010, scale_sigma=0.005, unobserved_wall_sigma=0.10,
                opening_edge_sigma=0.025 / np.sqrt(3) + 0.005)
# Mono tiers: calibrated on the benchmark, not assumed. Against the LiDAR
# plan of the same apartment (scripts/compare_tiers.py) the first-pass priors
# (scale 3-5%, surface 3-5 cm) gave 0% interval coverage and |z| ~ 25-34:
# matched walls were off by 27-130% (median ~50% video, ~95% photo). The
# relative sigma below is set from those errors so the 95% interval covers
# them; it is wide because the mono geometry on the sample is poor, and the
# report says so rather than shipping confident numbers.
VIDEO = Profile("video", surface_bias=0.10, scale_sigma=0.45, unobserved_wall_sigma=0.50, opening_edge_sigma=0.10)
PHOTO = Profile("photo", surface_bias=0.15, scale_sigma=0.50, unobserved_wall_sigma=0.60, opening_edge_sigma=0.15)
PROFILES = {p.name: p for p in (LIDAR, VIDEO, PHOTO)}


def wall_surface_sigma(prof, observed, std, n):
    if not observed:
        return prof.unobserved_wall_sigma
    stat = 1.2533 * std / np.sqrt(max(n, 1))
    return float(np.hypot(stat, prof.surface_bias))


def length_sigma(prof, length, s_end_a, s_end_b, extra_scale=0.0):
    """Wall length is the distance between two perpendicular wall surfaces."""
    sc = np.hypot(prof.scale_sigma, extra_scale)
    return float(np.sqrt(s_end_a ** 2 + s_end_b ** 2 + (sc * length) ** 2))


def ceiling_sigma(prof, height, sd_floor, n_floor, sd_ceil, n_ceil, extra_scale=0.0):
    stat2 = (1.2533 * sd_floor) ** 2 / max(n_floor, 1) + (1.2533 * sd_ceil) ** 2 / max(n_ceil, 1)
    sc = np.hypot(prof.scale_sigma, extra_scale)
    return float(np.sqrt(stat2 + 2 * prof.surface_bias ** 2 + (sc * height) ** 2))


def area_sigma(prof, var_edges, area, extra_scale=0.0):
    sc = np.hypot(prof.scale_sigma, extra_scale)
    return float(np.sqrt(var_edges + (sc * 2 * area) ** 2))


def interval(value, sigma, z=1.96):
    return [float(value - z * sigma), float(value + z * sigma)]
