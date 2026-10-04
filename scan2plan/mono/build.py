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


def _model_metric(m, depth_dir):
    """One SfM model scaled to (uncalibrated) metric by its own mono/SfM ratio."""
    names = sorted(m["images"])
    T = np.array([m["images"][n]["T_wc"] for n in names])
    Ks = [np.array(m["images"][n]["K"]) for n in names]
    sizes = [(m["images"][n]["width"], m["images"][n]["height"]) for n in names]
    ratios, per_image = [], []
    for n, K, (w, h) in zip(names, Ks, sizes):
        obs = np.array(m["images"][n]["obs"]).reshape(-1, 3)
        d = np.load(depth_dir / Path(n).with_suffix(".npy")).astype(np.float32)
        r = None
        if len(obs) >= 10:
            u = np.clip((obs[:, 0] / w * d.shape[1]).astype(int), 0, d.shape[1] - 1)
            v = np.clip((obs[:, 1] / h * d.shape[0]).astype(int), 0, d.shape[0] - 1)
            r = d[v, u] / obs[:, 2]
            r = r[np.isfinite(r) & (r > 0)]
            if len(r) < 10:
                r = None
        per_image.append(r)
        if r is not None:
            ratios.append(r)
    if not ratios:
        return None
    s = float(np.median(np.concatenate(ratios)))
    # Per-image correction k_i = s / median(mono/SfM) on that image.
    k = np.clip([s / np.median(r) if r is not None else 1.0 for r in per_image], 0.7, 1.4)
    T[:, :3, 3] *= s
    return dict(names=names, T=T, Ks=Ks, sizes=sizes, k=np.asarray(k, float),
                img_scales=[float(np.median(r)) for r in ratios], n_ratios=int(sum(len(r) for r in ratios)))


def _chain(models, depth_dir, features, min_inliers=40):
    """Attach models to the largest one through depth-lifted feature matches
    (see merge.py). Returns the merged model and a log of the links made."""
    from .merge import lift, ransac_rigid
    f = np.load(features)
    fnames = [str(n) for n in f["names"]]
    fidx = {n: i for i, n in enumerate(fnames)}
    pair_keys = [k for k in f.files if k.startswith("m_")]
    merged, rest = models[0], list(models[1:])
    where = {n: ("m", i) for i, n in enumerate(merged["names"])}
    links = []

    def cam_pts(model, i, kp_idx, fi):
        n = model["names"][i]
        d = np.load(depth_dir / Path(n).with_suffix(".npy")).astype(np.float32) * model["k"][i]
        P, ok = lift(f[f"kp_{fi}"][kp_idx], d, model["Ks"][i], model["sizes"][i])
        T = model["T"][i]
        return P @ T[:3, :3].T + T[:3, 3], ok

    while rest:
        best = None
        for ci, c in enumerate(rest):
            cidx = {n: i for i, n in enumerate(c["names"])}
            for key in pair_keys:
                a, b = map(int, key[2:].split("_"))
                na, nb = fnames[a], fnames[b]
                if na in where and nb in cidx:
                    im, ic, fm, fc, cols = where[na][1], cidx[nb], a, b, (0, 1)
                elif nb in where and na in cidx:
                    im, ic, fm, fc, cols = where[nb][1], cidx[na], b, a, (1, 0)
                else:
                    continue
                mt = f[key]
                Xm, okm = cam_pts(merged, im, mt[:, cols[0]], fm)
                Xc, okc = cam_pts(c, ic, mt[:, cols[1]], fc)
                ok = okm & okc
                M, ninl = ransac_rigid(Xc[ok], Xm[ok])
                if M is not None and ninl >= min_inliers and (best is None or ninl > best[0]):
                    best = (ninl, ci, M, merged["names"][im], c["names"][ic])
        if best is None:
            break
        ninl, ci, M, na, nb = best
        c = rest.pop(ci)
        # COLMAP may put one image in two models; keep its first placement.
        keep = [i for i, n in enumerate(c["names"]) if n not in where]
        c = dict(c, names=[c["names"][i] for i in keep], T=c["T"][keep], Ks=[c["Ks"][i] for i in keep],
                 sizes=[c["sizes"][i] for i in keep], k=c["k"][keep])
        Tc = M[None] @ c["T"]
        base = len(merged["names"])
        for key in ("names", "Ks", "sizes", "img_scales"):
            merged[key] = list(merged[key]) + list(c[key])
        merged["T"] = np.concatenate([merged["T"], Tc])
        merged["k"] = np.concatenate([merged["k"], c["k"]])
        merged["n_ratios"] += c["n_ratios"]
        for i, n in enumerate(c["names"]):
            where[n] = ("m", base + i)
        links.append(dict(images=len(c["names"]), via=[na, nb], inliers=ninl))
    return merged, links, [len(r["names"]) for r in rest]


def _up_from_points(T, names, Ks, sizes, k, depth_dir, up):
    """Resolve the sign of `up`: the scene (lifted mono depth) is mostly below
    the cameras for handheld capture."""
    heights = []
    for i in range(0, len(names), max(1, len(names) // 40)):
        d = np.load(depth_dir / Path(names[i]).with_suffix(".npy")).astype(np.float32) * k[i]
        H, W = d.shape
        K = np.array(Ks[i], float).copy()
        K[0] *= W / sizes[i][0]
        K[1] *= H / sizes[i][1]
        v, u = np.mgrid[0:H:8, 0:W:8]
        z = d[v, u]
        P = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1).reshape(-1, 3)
        Pw = P @ T[i, :3, :3].T + T[i, :3, 3]
        heights.append(np.median((Pw - T[i, :3, 3]) @ up))
    return up if np.median(heights) < 0 else -up


def load(sfm_json, depth_dir, image_root=None, features=None):
    """All SfM models -> one gravity-aligned, metric (pre-calibration) capture."""
    sfm = json.load(open(sfm_json))
    depth_dir = Path(depth_dir)
    models = [mm for mm in (_model_metric(m, depth_dir) for m in sfm["models"] if len(m["images"]) >= 3) if mm]
    if not models:
        raise RuntimeError("SfM registered no usable model")
    links, unplaced = [], [len(m["names"]) for m in models[1:]]
    if features is not None and len(models) > 1:
        merged, links, unplaced = _chain(models, depth_dir, features)
    else:
        merged = models[0]
    names, T, Ks, sizes, k = merged["names"], merged["T"], merged["Ks"], merged["sizes"], merged["k"]
    up, _ = gravity_from_cameras(T[:, :3, :3])
    up = _up_from_points(T, names, Ks, sizes, k, depth_dir, up)
    Rg = align_up(up)
    T[:, :3, :3] = Rg @ T[:, :3, :3]
    T[:, :3, 3] = T[:, :3, 3] @ Rg.T
    ls = np.log(merged["img_scales"])
    stats = dict(images_registered=len(names), images_total=len(sfm["images"]),
                 models=[len(m["images"]) for m in sfm["models"]], chained_links=links, unplaced_model_sizes=unplaced,
                 n_ratios=merged["n_ratios"],
                 per_image_scale_spread=float(np.std(ls)) if len(ls) > 1 else None,
                 scale_sigma_rel=float(1.4826 * np.median(np.abs(ls - np.median(ls))) / np.sqrt(max(len(ls), 1))))
    room_of = [str(Path(n).parent) if str(Path(n).parent) != "." else "" for n in names]
    order = np.argsort(names)
    cap = MonoCapture(Path(image_root or depth_dir), [names[i] for i in order], T[order], [Ks[i] for i in order],
                      [sizes[i] for i in order], depth_dir, k[order], [room_of[i] for i in order],
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


def exif_intrinsics(path, factor=0.83):
    """K from EXIF 35 mm-equivalent focal length when present, else the iPhone
    main-camera prior. 35 mm equivalence is defined on the diagonal."""
    from PIL import Image
    im = Image.open(path)
    w, h = im.size
    f = factor * max(w, h)
    try:
        ex = im.getexif().get_ifd(0x8769)
        f35 = ex.get(0xA405)
        if f35:
            f = float(f35) / 43.27 * np.hypot(w, h)
    except Exception:
        pass
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]), (w, h)


def load_depthgraph(img_dir, depth_dir, features):
    """Photo tier: poses from the depth-lifted pose graph (depthgraph.py)."""
    from . import depthgraph as DG
    img_dir, depth_dir = Path(img_dir), Path(depth_dir)
    f = np.load(features)
    names = [str(n) for n in f["names"]]
    KS = [exif_intrinsics(img_dir / n) for n in names]
    Ks, sizes = [k for k, _ in KS], [s for _, s in KS]
    depth_of = lambda i: np.load(depth_dir / Path(names[i]).with_suffix(".npy")).astype(np.float32)
    edges = DG.pair_edges(names, f, depth_of, Ks, sizes)
    tree, comps = DG.spanning_tree(len(names), edges)
    comp = comps[0]
    G = DG.compose(comp, tree)
    G = DG.refine(comp, G, edges)
    T, scale = [], []
    for k in comp:
        M = G[k][:3, :3]
        s = np.cbrt(np.linalg.det(M))
        P = np.eye(4)
        P[:3, :3] = M / s
        P[:3, 3] = G[k][:3, 3]
        T.append(P)
        scale.append(s)
    T, scale = np.array(T), np.array(scale)
    # Keep the network's metric scale on average: per-image scales are only
    # relative, so normalise their median to 1.
    m = np.median(scale)
    scale /= m
    T[:, :3, 3] /= m
    cn = [names[k] for k in comp]
    up, _ = gravity_from_cameras(T[:, :3, :3])
    up = _up_from_points(T, cn, [Ks[k] for k in comp], [sizes[k] for k in comp], scale, depth_dir, up)
    Rg = align_up(up)
    T[:, :3, :3] = Rg @ T[:, :3, :3]
    T[:, :3, 3] = T[:, :3, 3] @ Rg.T
    room_of = [str(Path(n).parent) if str(Path(n).parent) != "." else "" for n in cn]
    ls = np.log(scale)
    stats = dict(pose_source="depth graph (sim3 RANSAC on depth-lifted DISK+LightGlue matches)",
                 images_registered=len(comp), images_total=len(names), pair_edges=len(edges),
                 components=[len(c) for c in comps], rooms_placed=sorted(set(room_of)),
                 rooms_unplaced=sorted(set(str(Path(names[k]).parent) for c in comps[1:] for k in c) - set(room_of)),
                 per_image_scale_spread=float(np.std(ls)),
                 scale_sigma_rel=float(1.4826 * np.median(np.abs(ls - np.median(ls))) / np.sqrt(max(len(ls), 1))))
    cap = MonoCapture(img_dir, cn, T, [Ks[k] for k in comp], [sizes[k] for k in comp], depth_dir, scale, room_of,
                      np.arange(len(comp), dtype=float))
    return cap, stats
