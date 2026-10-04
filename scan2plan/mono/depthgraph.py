"""Pose graph from depth-lifted matches (photo tier; video fallback).

COLMAP needs baseline to triangulate; photos taken while turning on the spot
verify as degenerate and it registers almost nothing (3/40 on the sample
photo set). With a metric depth map per image, a match is a 3D-3D
correspondence and a pair's relative pose follows from three of them, with
or without baseline. Per pair: Umeyama similarity (R, t, s) by RANSAC; s
absorbs the network's per-image scale error. Global: maximum spanning tree
on inlier counts, composed from the root, then a joint refinement of all
edges (rotation, translation, log-scale) by robust least squares.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .merge import lift


def umeyama(A, B):
    """s, R, t minimising |s R A + t - B|."""
    ca, cb = A.mean(0), B.mean(0)
    A0, B0 = A - ca, B - cb
    U, S, Vt = np.linalg.svd(B0.T @ A0 / len(A))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / max(A0.var(0).sum(), 1e-12)
    return s, R, cb - s * R @ ca


def ransac_sim3(A, B, iters=400, rel_thr=0.06, abs_thr=0.04, seed=0):
    rng = np.random.default_rng(seed)
    n = len(A)
    if n < 8:
        return None
    thr = rel_thr * np.linalg.norm(B, axis=1) + abs_thr
    best = (0, None)
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        s, R, t = umeyama(A[idx], B[idx])
        if not 0.6 < s < 1.6:
            continue
        inl = np.linalg.norm(s * A @ R.T + t - B, axis=1) < thr
        if inl.sum() > best[0]:
            best = (inl.sum(), inl)
    if best[1] is None or best[0] < 8:
        return None
    s, R, t = umeyama(A[best[1]], B[best[1]])
    return dict(s=s, R=R, t=t, inliers=int(best[0]), ratio=float(best[0] / n))


def pair_edges(names, feats, depth_of, Ks, sizes, min_inliers=15):
    """Relative similarity per matched pair: maps camera-j points into camera i."""
    edges = []
    for key in [k for k in feats.files if k.startswith("m_")]:
        i, j = map(int, key[2:].split("_"))
        m = feats[key]
        Pi, oki = lift(feats[f"kp_{i}"][m[:, 0]], depth_of(i), Ks[i], sizes[i])
        Pj, okj = lift(feats[f"kp_{j}"][m[:, 1]], depth_of(j), Ks[j], sizes[j])
        ok = oki & okj
        e = ransac_sim3(Pj[ok], Pi[ok])
        if e and e["inliers"] >= min_inliers and e["ratio"] >= 0.25:
            edges.append((i, j, e))
    return edges


def spanning_tree(n, edges):
    """Maximum spanning tree (Prim) on inlier counts. Returns parent links per
    connected component and the components."""
    adj = {k: [] for k in range(n)}
    for i, j, e in edges:
        adj[i].append((e["inliers"], j, i, e, False))
        adj[j].append((e["inliers"], i, j, e, True))
    seen, comps, tree = set(), [], {}
    import heapq
    for root in sorted(range(n), key=lambda k: -sum(x[0] for x in adj[k])):
        if root in seen:
            continue
        comp, heap = [root], []
        seen.add(root)
        for x in adj[root]:
            heapq.heappush(heap, (-x[0], id(x), x))
        while heap:
            _, _, (w, child, parent, e, inv) = heapq.heappop(heap)
            if child in seen:
                continue
            seen.add(child)
            comp.append(child)
            tree[child] = (parent, e, inv)
            for x in adj[child]:
                if x[1] not in seen:
                    heapq.heappush(heap, (-x[0], id(x), x))
        comps.append(comp)
    return tree, sorted(comps, key=len, reverse=True)


def _edge_T(e, inv):
    """4x4 similarity mapping child-camera points into parent camera."""
    T = np.eye(4)
    T[:3, :3] = e["s"] * e["R"]
    T[:3, 3] = e["t"]
    return np.linalg.inv(T) if inv else T


def compose(comp, tree):
    """World = root camera. Returns {k: 4x4 similarity camera_k -> world}."""
    G = {comp[0]: np.eye(4)}
    for k in comp[1:]:
        chain, c = [], k
        while c not in G:
            p, e, inv = tree[c]
            chain.append((c, p, e, inv))
            c = p
        for c, p, e, inv in reversed(chain):
            G[c] = G[p] @ _edge_T(e, inv)
    return G


def refine(comp, G, edges):
    """Joint least squares over all edges inside the component."""
    idx = {k: n for n, k in enumerate(comp)}
    E = [(i, j, e) for i, j, e in edges if i in idx and j in idx]

    def unpack(x):
        out = {}
        for k, n in idx.items():
            r, t, ls = x[7 * n:7 * n + 3], x[7 * n + 3:7 * n + 6], x[7 * n + 6]
            out[k] = (Rotation.from_rotvec(r).as_matrix(), t, np.exp(ls))
        return out

    x0 = []
    for k in comp:
        M = G[k][:3, :3]
        s = np.cbrt(np.linalg.det(M))
        x0 += list(Rotation.from_matrix(M / s).as_rotvec()) + list(G[k][:3, 3]) + [np.log(s)]
    x0 = np.array(x0)

    def res(x):
        P = unpack(x)
        r = []
        # Anchor the root (gauge freedom).
        R0, t0, s0 = P[comp[0]]
        r += list(Rotation.from_matrix(R0).as_rotvec() * 100) + list(t0 * 100) + [np.log(s0) * 100]
        for i, j, e in E:
            Ri, ti, si = P[i]
            Rj, tj, sj = P[j]
            # Predicted relative (j -> i) vs measured.
            Rp = Ri.T @ Rj
            sp = sj / si
            tp = Ri.T @ (tj - ti) / si
            w = np.sqrt(e["inliers"]) / 5
            r += list(Rotation.from_matrix(Rp @ e["R"].T).as_rotvec() * 5 * w)
            r += list((tp - e["t"]) * w)
            r += [np.log(sp / e["s"]) * 3 * w]
        return np.array(r)

    sol = least_squares(res, x0, loss="huber", f_scale=1.0, max_nfev=200)
    P = unpack(sol.x)
    out = {}
    for k, (R, t, s) in P.items():
        T = np.eye(4)
        T[:3, :3] = s * R
        T[:3, 3] = t
        out[k] = T
    return out
