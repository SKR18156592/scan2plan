"""Chain disconnected SfM models into one metric frame.

COLMAP needs many well-spread matches to join two image sets; in low-texture
rooms it often returns several models. Two images from different models
frequently still share 15-50 LightGlue matches. With metric depth, each
match becomes a 3D-3D correspondence, and three are enough for a rigid
transform: RANSAC (Kabsch) on those gives the relative pose of the two
cameras, hence of the two models. Models are attached greedily, best-supported
link first, to the growing merged set.
"""
import numpy as np


def kabsch(A, B):
    """R, t minimising |R A + t - B|."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def ransac_rigid(A, B, thr=0.10, iters=500, seed=0):
    rng = np.random.default_rng(seed)
    best = (0, None)
    n = len(A)
    if n < 6:
        return None, 0
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        R, t = kabsch(A[idx], B[idx])
        inl = np.linalg.norm(A @ R.T + t - B, axis=1) < thr
        if inl.sum() > best[0]:
            best = (inl.sum(), inl)
    if best[1] is None or best[0] < 6:
        return None, 0
    R, t = kabsch(A[best[1]], B[best[1]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T, int(best[0])


def lift(kp, depth, K, size):
    """Keypoints (original pixels) -> camera-frame 3D using a depth map."""
    w, h = size
    H, W = depth.shape
    u = np.clip((kp[:, 0] / w * W).astype(int), 0, W - 1)
    v = np.clip((kp[:, 1] / h * H).astype(int), 0, H - 1)
    z = depth[v, u]
    x = (kp[:, 0] - K[0, 2]) / K[0, 0] * z
    y = (kp[:, 1] - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], 1), (z > 0.2) & (z < 6)
