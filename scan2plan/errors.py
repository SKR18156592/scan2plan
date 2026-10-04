"""Error model for the LiDAR tier. Every reported measurement gets a 1-sigma
value from here and is published as a 95% interval (value +- 1.96 sigma).

The constants are priors, not fits: there is no ground truth for the sample
captures. They are the numbers to recalibrate once tape/laser measurements
exist (see report: calibration analysis).
"""
import numpy as np

# Per-surface position bias of iPhone LiDAR depth after averaging many frames.
# Single-frame depth noise is ~1-2 cm; averaging removes noise but not the
# bias of the depth sensor/ARKit fusion, which public evaluations put at ~1 cm.
LIDAR_SURFACE_BIAS = 0.010

# Relative scale error of ARKit visual-inertial odometry over a room-sized
# path. Applies to every length as a fraction of that length.
SCALE_SIGMA = 0.005

# A wall edge with no supporting points keeps the room-mask boundary, which
# is only good to about two grid cells plus the wall dilation.
UNOBSERVED_WALL_SIGMA = 0.10

# Opening edges are found on a 2.5 cm sampling grid along the wall.
OPENING_EDGE_SIGMA = 0.025 / np.sqrt(3) * np.sqrt(2) + 0.005


def length_sigma(length, s_end_a, s_end_b):
    """Wall length is the distance between two perpendicular wall surfaces."""
    return float(np.sqrt(s_end_a ** 2 + s_end_b ** 2 + (SCALE_SIGMA * length) ** 2))


def ceiling_sigma(height, sd_floor, n_floor, sd_ceil, n_ceil):
    stat2 = (1.2533 * sd_floor) ** 2 / max(n_floor, 1) + (1.2533 * sd_ceil) ** 2 / max(n_ceil, 1)
    return float(np.sqrt(stat2 + 2 * LIDAR_SURFACE_BIAS ** 2 + (SCALE_SIGMA * height) ** 2))


def interval(value, sigma, z=1.96):
    return [float(value - z * sigma), float(value + z * sigma)]
