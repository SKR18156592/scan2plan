"""Fuse per-frame LiDAR depth into one world-frame point cloud."""
import numpy as np

from .io import backproject


def voxel_downsample(pts, voxel):
    """Average points per voxel. Returns (centroids, counts)."""
    keys = np.floor(pts / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((len(counts), 3))
    np.add.at(out, inv, pts)
    return out / counts[:, None], counts


def fuse(cap, poses=None, step=3, voxel=0.02, max_depth=4.0, chunk=200):
    """Back-project every `step`-th frame and voxel-downsample.

    Depth beyond ~4 m on the iPhone LiDAR is sparse and noisy, so it is cut.
    `poses` overrides the capture's own poses (used for drift correction).
    """
    if poses is not None:
        cap = _with_poses(cap, poses)
    parts, weights = [], []
    frames = list(range(0, len(cap), step))
    for s in range(0, len(frames), chunk):
        pts = np.concatenate([backproject(cap, i, max_depth=max_depth) for i in frames[s:s + chunk]])
        p, c = voxel_downsample(pts, voxel)
        parts.append(p * c[:, None])
        weights.append(c)
    # Merge chunk results: re-voxelise the count-weighted sums.
    sums = np.concatenate(parts)
    w = np.concatenate(weights)
    centers = sums / w[:, None]
    keys = np.floor(centers / voxel).astype(np.int64)
    _, inv, _ = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    n = inv.max() + 1
    acc = np.zeros((n, 3))
    cnt = np.zeros(n)
    np.add.at(acc, inv, sums)
    np.add.at(cnt, inv, w)
    return acc / cnt[:, None], cnt


def _with_poses(cap, poses):
    from dataclasses import replace
    return replace(cap, poses=poses)
