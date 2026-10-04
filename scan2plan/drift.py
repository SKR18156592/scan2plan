"""Drift correction: fragment loop closure + 4-DOF pose graph.

ARKit odometry is gravity-aligned, so roll and pitch do not drift; what
accumulates is yaw and horizontal translation. Height is anchored by the
floor plane, so the graph is effectively 3-DOF (x, z, yaw). The trajectory is cut into short
fragments (each locally accurate). Fragments that revisit the same place
later in the capture are aligned with point-to-plane ICP; each accepted
alignment is a loop-closure edge. A pose graph over per-fragment corrections
(tx, tz, yaw) is solved with odometry edges (consecutive fragments keep
their relative pose) and loop edges, then corrections are interpolated to
every frame.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .cloud import voxel_downsample
from .io import backproject


def _rot_y(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _T(p):
    """4-DOF params (tx, ty, tz, yaw) -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = _rot_y(p[3])
    T[:3, 3] = p[:3]
    return T


def _params(T):
    yaw = np.arctan2(T[0, 2], T[0, 0])
    return np.array([T[0, 3], T[1, 3], T[2, 3], yaw])


def fragments(cap, size=90, sub=5, voxel=0.03):
    """Frame ranges and their world-frame point clouds."""
    frags = []
    for s in range(0, len(cap), size):
        idx = range(s, min(s + size, len(cap)), sub)
        pts = np.concatenate([backproject(cap, i, max_depth=3.5) for i in idx])
        if len(pts) < 2000:
            frags.append(None)
            continue
        p, _ = voxel_downsample(pts, voxel)
        cam = cap.poses[s:s + size, :3, 3]
        frags.append(dict(start=s, end=min(s + size, len(cap)), pts=p, center=cam.mean(0)))
    return frags


def _normals(pts, tree, k=12):
    _, nn = tree.query(pts, k=k)
    nb = pts[nn] - pts[nn].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    w, v = np.linalg.eigh(cov)
    return v[:, :, 0]


def icp(src, dst, dst_tree, dst_n, iters=30, max_dist=0.15):
    """Point-to-plane ICP in the floor plane: (tx, tz, yaw), ty fixed at 0.

    Height is anchored by the floor plane (every fragment sees the floor and
    ARKit's vertical is gravity-referenced), so ICP is not allowed to slide
    fragments vertically. Only wall points (horizontal normals) are passed in.

    Returns (T, fitness, rmse, min_eig_ratio). The eigenvalue ratio of the
    3x3 normal matrix measures how well-constrained the alignment is; a
    corridor (two parallel walls) leaves translation along it unconstrained.
    """
    p = np.zeros(4)
    H = np.eye(3)
    for it in range(iters):
        T = _T(p)
        s = src @ T[:3, :3].T + T[:3, 3]
        d, j = dst_tree.query(s, distance_upper_bound=max_dist)
        m = np.isfinite(d)
        if m.sum() < 100:
            return None, 0.0, np.inf, 0.0
        q, n, sm = dst[j[m]], dst_n[j[m]], s[m]
        r = np.einsum("ij,ij->i", sm - q, n)
        c = sm - T[:3, 3]
        dyaw = np.stack([c[:, 2], np.zeros(len(c)), -c[:, 0]], 1)
        J = np.c_[n[:, 0], n[:, 2], np.einsum("ij,ij->i", dyaw, n)]
        w = 1.0 / np.maximum(1.0, np.abs(r) / 0.02)
        H = J.T @ (J * w[:, None])
        g = J.T @ (w * r)
        dp = -np.linalg.solve(H + 1e-6 * np.eye(3), g)
        p[[0, 2, 3]] += dp
        if np.abs(dp[:2]).max() < 1e-4 and abs(dp[2]) < 1e-5:
            break
        max_dist = max(0.05, max_dist * 0.85)
    T = _T(p)
    s = src @ T[:3, :3].T + T[:3, 3]
    d, _ = dst_tree.query(s, distance_upper_bound=0.05)
    m = np.isfinite(d)
    # Conditioning of the translation block only (yaw is scaled differently).
    ev = np.linalg.eigvalsh(H[:2, :2])
    ratio = float(ev[0] / max(ev[1], 1e-12))
    return T, float(m.mean()), float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else np.inf, ratio


def _wall_points(pts, k=12):
    tree = cKDTree(pts)
    n = _normals(pts, tree, k)
    keep = np.abs(n[:, 1]) < 0.3
    return pts[keep], n[keep]


def loop_edges(frags, min_gap=8, radius=1.5, min_fitness=0.35, min_cond=0.1, log=None):
    valid = [k for k, f in enumerate(frags) if f is not None]
    walls = {}
    edges = []
    for a in valid:
        for b in valid:
            if b - a < min_gap:
                continue
            if np.linalg.norm(frags[a]["center"] - frags[b]["center"]) > radius:
                continue
            for k in (a, b):
                if k not in walls:
                    wp, wn = _wall_points(frags[k]["pts"])
                    walls[k] = (wp, wn, cKDTree(wp) if len(wp) else None)
            dst, dst_n, tree = walls[a]
            src = walls[b][0]
            if len(src) < 500 or len(dst) < 500:
                continue
            if len(src) > 15000:
                src = src[np.random.default_rng(b).choice(len(src), 15000, replace=False)]
            T, fit, rmse, cond = icp(src, dst, tree, dst_n)
            if T is None:
                continue
            p = _params(T)
            ok = (fit >= min_fitness and cond >= min_cond and np.linalg.norm(p[:3]) < 0.6
                  and abs(np.rad2deg(p[3])) < 8)
            if log is not None:
                log.append(dict(a=a, b=b, fitness=fit, rmse=rmse, cond=cond, t=p[:3].tolist(),
                                yaw_deg=float(np.rad2deg(p[3])), accepted=bool(ok)))
            if ok:
                edges.append((a, b, T, fit))
    return edges


# Noise model (1 sigma). Odometry: drift accumulated across one ~2 s
# fragment boundary. Loop: ICP alignment error between two fragments.
ODO_SIGMA = np.array([0.01, 1e-4, 0.01, np.deg2rad(0.1)])
LOOP_SIGMA = np.array([0.02, 1e-4, 0.02, np.deg2rad(0.3)])


# Manhattan heading prior: per-fragment wall direction is measured to ~0.5 deg.
MANHATTAN_SIGMA = np.deg2rad(0.5)


def _wrap90(a):
    q = np.pi / 2
    return (a + q / 2) % q - q / 2


def fragment_headings(frags, min_pts=1500):
    """Dominant wall direction (rad, mod 90 deg) of each fragment, or nan.

    In a rectilinear building every fragment's walls should share one
    direction; a fragment whose walls are rotated by d has accumulated
    heading drift d. This is the plane-anchored correction: it constrains
    yaw everywhere, not only where the path happens to loop.
    """
    from .layout import dominant_angle
    out = np.full(len(frags), np.nan)
    for k, f in enumerate(frags):
        if f is None:
            continue
        wp, _ = _wall_points(f["pts"])
        if len(wp) >= min_pts:
            out[k] = dominant_angle(wp[:, [0, 2]])
    return out


def optimize(n, edges, headings=None):
    """Per-fragment corrections C_k (4-DOF). C_0 fixed at identity.

    Odometry edge: C_k and C_{k+1} should be equal (consecutive fragments
    are consistent; drift is what accumulates between distant ones).
    Loop edge (a, b, T): T maps fragment b's points onto fragment a's, so
    C_b should equal C_a @ T. Residuals are whitened by their sigma and a
    Huber loss (3 sigma) limits the pull of any wrong loop closure.
    """
    def res(x):
        P = np.vstack([np.zeros(4), x.reshape(-1, 4)])
        r = [((P[1:] - P[:-1]) / ODO_SIGMA).ravel()]
        for a, b, T, fit in edges:
            target = _params(_T(P[a]) @ T)
            d = P[b] - target
            d[3] = (d[3] + np.pi) % (2 * np.pi) - np.pi
            r.append(d / LOOP_SIGMA)
        if headings is not None:
            ok = np.isfinite(headings)
            # Rotating a fragment by +yaw lowers its measured wall angle by yaw.
            r.append(_wrap90(headings[ok] - P[ok, 3] - ref) / MANHATTAN_SIGMA)
        return np.concatenate(r)

    if headings is not None:
        h = headings[np.isfinite(headings)]
        # Reference direction: circular median of the fragment headings (mod 90).
        ref = h[0] + np.median(_wrap90(h - h[0]))

    x0 = np.zeros((n - 1) * 4)
    sol = least_squares(res, x0, loss="huber", f_scale=3.0)
    P = np.vstack([np.zeros(4), sol.x.reshape(-1, 4)])
    loop_res = []
    for a, b, T, fit in edges:
        before = np.linalg.norm(_params(T)[:3])
        after = np.linalg.norm((P[b] - _params(_T(P[a]) @ T))[:3])
        loop_res.append((before, after))
    return P, np.array(loop_res)


def correct(cap, log=None, manhattan=True):
    """Return drift-corrected camera poses (N, 4, 4) and a summary dict."""
    frags = fragments(cap)
    edges = loop_edges(frags, log=log)
    headings = fragment_headings(frags) if manhattan else None
    C, loop_res = optimize(len(frags), edges, headings)
    centers = np.array([(f["start"] + f["end"]) / 2 if f else k * 90 + 45 for k, f in enumerate(frags)])
    frame_idx = np.arange(len(cap))
    Cf = np.stack([np.interp(frame_idx, centers, C[:, j]) for j in range(4)], 1)
    poses = np.array([_T(Cf[i]) @ cap.poses[i] for i in frame_idx])
    if headings is not None:
        ok = np.isfinite(headings)
        ref = headings[ok][0] + np.median(_wrap90(headings[ok] - headings[ok][0]))
        before = np.rad2deg(_wrap90(headings[ok] - ref))
        after = np.rad2deg(_wrap90(headings[ok] - C[ok, 3] - ref))
    summary = dict(method="fragment ICP loop closure + Manhattan heading prior, pose graph (x, z, yaw)"
                   if manhattan else "fragment ICP loop closure, pose graph (x, z, yaw)",
                   heading_spread_deg=dict(before=float(before.std()), after=float(after.std()),
                                           fragments_used=int(ok.sum())) if headings is not None else None,
                   fragments=len(frags), loop_edges=len(edges),
                   max_correction_m=float(np.linalg.norm(C[:, :3], axis=1).max()),
                   max_yaw_correction_deg=float(np.rad2deg(np.abs(C[:, 3]).max())),
                   loop_residual_before_m=dict(mean=float(loop_res[:, 0].mean()), max=float(loop_res[:, 0].max())) if len(loop_res) else None,
                   loop_residual_after_m=dict(mean=float(loop_res[:, 1].mean()), max=float(loop_res[:, 1].max())) if len(loop_res) else None,
                   end_correction=dict(t=C[-1, :3].tolist(), yaw_deg=float(np.rad2deg(C[-1, 3]))))
    return poses, summary
