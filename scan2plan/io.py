"""Loader for Stray Scanner LiDAR captures.

Layout: rgb.mp4, depth/NNNNNN.png (uint16 mm, 256x192), confidence/NNNNNN.png
(0/1/2), odometry.csv (camera-to-world pose per frame; Stray already
exports camera axes in OpenCV convention: x right, y down, z forward, world y up),
camera_matrix.csv (intrinsics at RGB resolution), imu.csv.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

@dataclass
class Capture:
    root: Path
    timestamps: np.ndarray  # (N,)
    poses: np.ndarray  # (N, 4, 4) camera-to-world, OpenCV camera axes
    K_rgb: np.ndarray  # (3, 3) intrinsics at RGB resolution
    rgb_size: tuple  # (w, h)

    def __len__(self):
        return len(self.timestamps)

    def depth(self, i):
        """Depth in metres at depth-map resolution."""
        return np.asarray(Image.open(self.root / "depth" / f"{i:06d}.png"), np.float32) / 1000.0

    def confidence(self, i):
        return np.asarray(Image.open(self.root / "confidence" / f"{i:06d}.png"))

    def K_depth(self, shape, i=None):
        h, w = shape
        K = self.K_rgb.copy()
        K[0] *= w / self.rgb_size[0]
        K[1] *= h / self.rgb_size[1]
        return K


def load_capture(root, rgb_size=(1920, 1440)):
    root = Path(root)
    odo = np.genfromtxt(root / "odometry.csv", delimiter=",", skip_header=1, usecols=range(9))
    t, xyz, quat = odo[:, 0], odo[:, 2:5], odo[:, 5:9]
    poses = np.tile(np.eye(4), (len(t), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(quat).as_matrix()
    poses[:, :3, 3] = xyz
    K = np.loadtxt(root / "camera_matrix.csv", delimiter=",")
    return Capture(root, t, poses, K, rgb_size)


def backproject(cap, i, min_conf=2, max_depth=5.0):
    """World-frame points (M, 3) from frame i, high-confidence pixels only."""
    d = cap.depth(i)
    c = cap.confidence(i)
    K = cap.K_depth(d.shape, i)
    v, u = np.nonzero((c >= min_conf) & (d > 0.1) & (d < max_depth))
    z = d[v, u]
    pts = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
    T = cap.poses[i]
    return pts @ T[:3, :3].T + T[:3, 3]
