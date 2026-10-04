"""Turn SfM poses + monocular depth into a metric, gravity-aligned capture.

SfM gives poses up to an unknown scale and an arbitrary world frame. The
metric depth network gives a scale per image but its depth is only good to
~10%. Combining them:

  scale    s   = median over images and sparse points of mono / SfM depth.
                 One number for the whole model, so per-image network errors
                 average out.
  per-image k_i = median(s * SfM depth / mono depth) on that image's points.
                 Corrects each depth map's own scale against the shared
                 geometry before fusion.
  gravity      : handheld captures keep roll near zero, so camera x-axes lie
                 in the horizontal plane; gravity is the normal of that plane
                 (smallest eigenvector of sum x x^T). Then refined by the floor.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class MonoCapture:
    """Same interface as io.Capture, so fuse/layout/rooms work unchanged."""
    root: Path
    names: list
    poses: np.ndarray  # (N, 4, 4) camera-to-world, metric, y up
    Ks: list  # per-image intrinsics at original image resolution
    sizes: list  # per-image (w, h)
    depth_dir: Path
    depth_scale: np.ndarray  # per-image k_i
    room_of: list = field(default_factory=list)  # photo tier: folder name per image
    timestamps: np.ndarray = None

    def __len__(self):
        return len(self.names)

    def depth(self, i):
        d = np.load(self.depth_dir / Path(self.names[i]).with_suffix(".npy")).astype(np.float32)
        return d * self.depth_scale[i]

    def confidence(self, i):
        """Mono depth has no confidence; mark depth discontinuities as low
        confidence (they produce the same flying pixels as LiDAR edges)."""
        d = self.depth(i)
        gy, gx = np.gradient(d)
        rel = np.hypot(gx, gy) / np.maximum(d, 0.1)
        c = np.full(d.shape, 2, np.uint8)
        c[rel > 0.05] = 0
        # The image border is the least reliable part of the prediction.
        c[:4], c[-4:], c[:, :4], c[:, -4:] = 0, 0, 0, 0
        return c

    def K_depth(self, shape, i=None):
        K = np.array(self.Ks[i if i is not None else 0], float).copy()
        w, h = self.sizes[i if i is not None else 0]
        K[0] *= shape[1] / w
        K[1] *= shape[0] / h
        return K


def gravity_from_cameras(R_wc):
    """Up vector (sign unresolved) and the image axis that is horizontal.

    A handheld phone keeps one image axis level (x in landscape, y in
    portrait, roll ~ 0), so that axis sweeps a horizontal plane across the
    capture; gravity is that plane's normal. Whichever of x/y is more
    coplanar (smaller eigenvalue) is the level one. Stray Scanner stores
    portrait captures sideways, so this cannot be assumed.
    """
    best = None
    for ax in (0, 1):
        a = R_wc[:, :, ax]
        w, v = np.linalg.eigh(a.T @ a)
        score = w[0] / max(w.sum(), 1e-12)
        if best is None or score < best[0]:
            best = (score, v[:, 0], ax)
    return best[1], best[2]


def image_up_rotation(R_wc, up):
    """Clockwise quarter-turns that make the images upright, given world up.

    The image direction (-y, +x, +y, -x) best aligned with up is "image up";
    returns k such that np.rot90(img, -k) is upright.
    """
    dirs = [-R_wc[:, :, 1], R_wc[:, :, 0], R_wc[:, :, 1], -R_wc[:, :, 0]]
    j = int(np.argmax([np.median(d @ up) for d in dirs]))
    # Edge j must move to the top: top 0 turns, right 3, bottom 2, left 1.
    return [0, 3, 2, 1][j]


def align_up(up):
    """Rotation taking `up` to +y."""
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(up, y)
    c = float(np.dot(up, y))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def sfm_up(model):
    """Gravity in SfM world coordinates, with the sign resolved.

    Sign: handheld captures look down more than up (median pitch on the
    sample captures is 19-31 deg), so the sparse points are mostly below the
    cameras; up is the direction that puts the median point below them.
    """
    names = sorted(model["images"])
    T = np.array([model["images"][n]["T_wc"] for n in names])
    up, _ = gravity_from_cameras(T[:, :3, :3])
    pts = []
    for n in names:
        im = model["images"][n]
        obs = np.array(im["obs"]).reshape(-1, 3)
        if len(obs) == 0:
            continue
        K = np.array(im["K"])
        rays = np.c_[(obs[:, 0] - K[0, 2]) / K[0, 0], (obs[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(obs))] * obs[:, 2:3]
        Tw = np.array(im["T_wc"])
        pts.append(rays @ Tw[:3, :3].T + Tw[:3, 3])
    pts = np.concatenate(pts)
    if np.median(pts @ up) > np.median(T[:, :3, 3] @ up):
        up = -up
    return up


def orientation(sfm_json, model_index=0):
    """Quarter-turns needed to show this capture's images upright."""
    m = json.load(open(sfm_json))["models"][model_index]
    names = sorted(m["images"])
    T = np.array([m["images"][n]["T_wc"] for n in names])
    return image_up_rotation(T[:, :3, :3], sfm_up(m))


def load(sfm_json, depth_dir, model_index=0, image_root=None):
    sfm = json.load(open(sfm_json))
    if not sfm["models"]:
        raise RuntimeError("SfM registered no images")
    m = sfm["models"][model_index]
    names = sorted(m["images"])
    depth_dir = Path(depth_dir)
    T = np.array([m["images"][n]["T_wc"] for n in names])
    Ks = [np.array(m["images"][n]["K"]) for n in names]
    sizes = [(m["images"][n]["width"], m["images"][n]["height"]) for n in names]

    # Global scale from sparse points.
    ratios, per_image = [], []
    for n, K, (w, h) in zip(names, Ks, sizes):
        obs = np.array(m["images"][n]["obs"]).reshape(-1, 3)
        d = np.load(depth_dir / Path(n).with_suffix(".npy")).astype(np.float32)
        if len(obs) < 10:
            per_image.append(None)
            continue
        u = np.clip((obs[:, 0] / w * d.shape[1]).astype(int), 0, d.shape[1] - 1)
        v = np.clip((obs[:, 1] / h * d.shape[0]).astype(int), 0, d.shape[0] - 1)
        r = d[v, u] / obs[:, 2]
        r = r[np.isfinite(r) & (r > 0)]
        if len(r) >= 10:
            ratios.append(r)
            per_image.append(r)
        else:
            per_image.append(None)
    allr = np.concatenate(ratios)
    s = float(np.median(allr))
    img_scales = np.array([np.median(r) for r in ratios])
    # Per-image correction k_i = s / median(mono/SfM) on that image.
    k = np.array([s / np.median(r) if r is not None else 1.0 for r in per_image])
    k = np.clip(k, 0.7, 1.4)

    T[:, :3, 3] *= s
    up = sfm_up(m)
    Rg = align_up(up)
    T[:, :3, :3] = Rg @ T[:, :3, :3]
    T[:, :3, 3] = T[:, :3, 3] @ Rg.T
    stats = dict(scale=s, n_ratios=int(len(allr)), images_registered=len(names),
                 images_total=len(sfm["images"]),
                 per_image_scale_spread=float(np.std(np.log(img_scales))) if len(img_scales) > 1 else None,
                 scale_sigma_rel=float(1.4826 * np.median(np.abs(np.log(img_scales) - np.median(np.log(img_scales)))) / np.sqrt(max(len(img_scales), 1))))
    room_of = [str(Path(n).parent) if str(Path(n).parent) != "." else "" for n in names]
    cap = MonoCapture(Path(image_root or depth_dir), names, T, Ks, sizes, depth_dir, k, room_of,
                      np.arange(len(names), dtype=float))
    return cap, stats


def refine_gravity_with_floor(cap, pts):
    """Tilt the world so the floor plane is exactly horizontal.

    The camera-axis gravity estimate is off by the average roll of the
    handheld phone (typically 1-3 deg). The floor is the lowest large plane
    roughly perpendicular to that estimate: RANSAC on the lowest points.
    """
    rng = np.random.default_rng(0)
    y = pts[:, 1]
    low = pts[y < np.percentile(y, 25)]
    if len(low) < 500:
        return cap, np.eye(3), pts
    best = (0, None)
    for _ in range(300):
        p = low[rng.choice(len(low), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n[1]) < np.cos(np.deg2rad(15)):
            continue
        inl = np.abs((low - p[0]) @ n) < 0.03
        if inl.sum() > best[0]:
            best = (inl.sum(), n)
    if best[1] is None:
        return cap, np.eye(3), pts
    n = best[1] if best[1][1] > 0 else -best[1]
    R = align_up(n)
    cap.poses[:, :3, :3] = R @ cap.poses[:, :3, :3]
    cap.poses[:, :3, 3] = cap.poses[:, :3, 3] @ R.T
    return cap, R, pts @ R.T
