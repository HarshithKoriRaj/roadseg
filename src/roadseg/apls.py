"""APLS (Average Path Length Similarity) for rasterised road predictions.

Ported from sam2_road_dac with two corrections:

1. Nearest-node lookup uses a cKDTree. The original scanned every node per
   query, making the metric O(N^2) on graphs with thousands of nodes.
2. A `max_snap_dist` threshold is enforced. The original snapped to the nearest
   node at any distance, so a prediction whose nearest node was far away still
   received credit; the standard metric scores an unmatched node 0. Without
   this the reported APLS is systematically inflated.

Shortest paths are computed with batched scipy.sparse.csgraph.dijkstra rather
than one networkx call per sampled pair.

Conventions: if both graphs are empty (no roads, none predicted) the score is
1.0; if exactly one is empty it is 0.0. Pairs are sampled uniformly over graph
NODES, not over road length, so axis-aligned roads on the sampling lattice
carry more nodes per unit length than diagonal ones. This affects prediction
and ground truth alike.
"""
from __future__ import annotations

import heapq
import math

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


_NBRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def graph_from_mask(mask: np.ndarray, spacing: int = 8) -> nx.Graph:
    """Skeletonise a binary mask into a graph.

    Connectivity comes from the skeleton itself (8-connected pixels), so it
    cannot join separate roads that merely lie close together, nor split one
    road whose samples happen to be sparse. Nodes are a thinning of the
    skeleton pixels: every pixel with y or x divisible by `spacing`, plus all
    endpoints and junctions (pixel degree != 2), plus one pixel for any
    component that would otherwise have none. Two nodes are linked when a
    skeleton path joins them without passing through another node; the edge
    `length` is that path's length in pixels.
    """
    sk = skeletonize(mask.astype(bool))
    ys, xs = np.nonzero(sk)
    if len(ys) == 0:
        return nx.Graph()
    pix = {(int(y), int(x)) for y, x in zip(ys, xs)}

    def nbrs(p):
        y, x = p
        for dy, dx in _NBRS:
            q = (y + dy, x + dx)
            if q in pix:
                yield q, (math.sqrt(2.0) if dy and dx else 1.0)

    keep = {p for p in pix if p[0] % spacing == 0 or p[1] % spacing == 0
            or sum(1 for _ in nbrs(p)) != 2}
    # every connected skeleton component needs at least one node
    seen: set = set()
    for p in sorted(pix):
        if p in seen:
            continue
        comp, stack = {p}, [p]
        while stack:
            for q, _ in nbrs(stack.pop()):
                if q not in comp:
                    comp.add(q); stack.append(q)
        seen |= comp
        if not (comp & keep):
            keep.add(p)

    order = sorted(keep)
    idx = {p: i for i, p in enumerate(order)}
    G = nx.Graph()
    for p, i in idx.items():
        G.add_node(i, y=float(p[0]), x=float(p[1]))
    for src in order:
        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, p = heapq.heappop(heap)
            if d > dist.get(p, math.inf):
                continue
            if p in keep and p != src:
                a, b = idx[src], idx[p]
                if not G.has_edge(a, b) or G[a][b]["length"] > d:
                    G.add_edge(a, b, length=d)
                continue                      # do not walk through other nodes
            for q, w in nbrs(p):
                nd = d + w
                if nd < dist.get(q, math.inf):
                    dist[q] = nd
                    heapq.heappush(heap, (nd, q))
    return G


def _pos_array(G: nx.Graph) -> tuple[np.ndarray, list]:
    nodes = list(G.nodes())
    pts = np.array([[G.nodes[n]["y"], G.nodes[n]["x"]] for n in nodes], dtype=float)
    return pts, nodes


def _ensure_edge_lengths(G: nx.Graph) -> None:
    for u, v, data in G.edges(data=True):
        if "length" not in data:
            a, b = G.nodes[u], G.nodes[v]
            data["length"] = math.hypot(a["y"] - b["y"], a["x"] - b["x"])


def _csr(G: nx.Graph, nodes: list) -> csr_matrix:
    index = {n: i for i, n in enumerate(nodes)}
    m = G.number_of_edges()
    r = np.empty(m, dtype=np.int64); c = np.empty(m, dtype=np.int64)
    w = np.empty(m, dtype=float)
    for k, (u, v, d) in enumerate(G.edges(data=True)):
        r[k], c[k], w[k] = index[u], index[v], d["length"]
    n = len(nodes)
    return csr_matrix((w, (r, c)), shape=(n, n))


def _pair_lengths(csr: csr_matrix, s: np.ndarray, t: np.ndarray,
                  max_bytes: int = 200_000_000) -> np.ndarray:
    """Shortest-path length for each (s[k], t[k]); inf if unreachable."""
    out = np.empty(len(s), dtype=float)
    if len(s) == 0:
        return out
    uniq, inv = np.unique(s, return_inverse=True)
    chunk = max(1, int(max_bytes // (8 * csr.shape[0])))
    for lo in range(0, len(uniq), chunk):
        D = dijkstra(csr, directed=False, indices=uniq[lo:lo + chunk])
        sel = np.nonzero((inv >= lo) & (inv < lo + chunk))[0]
        out[sel] = D[inv[sel] - lo, t[sel]]
    return out


def _one_way(G_src: nx.Graph, G_dst: nx.Graph, sample: int, rng, max_snap_dist: float) -> float:
    src_nodes = list(G_src.nodes())
    if len(src_nodes) < 2 or G_dst.number_of_nodes() == 0:
        return 0.0
    dst_pts, dst_nodes = _pos_array(G_dst)
    src_pts, _ = _pos_array(G_src)
    tree = cKDTree(dst_pts)
    pairs = np.array([rng.choice(len(src_nodes), size=2, replace=False)
                      for _ in range(sample)], dtype=np.int64)
    si, ti = pairs[:, 0], pairs[:, 1]
    L1 = _pair_lengths(_csr(G_src, src_nodes), si, ti)
    ok = np.isfinite(L1) & (L1 > 0)
    if not ok.any():
        return 0.0
    si, ti, L1 = si[ok], ti[ok], L1[ok]
    d_s, i_s = tree.query(src_pts[si], k=1)
    d_t, i_t = tree.query(src_pts[ti], k=1)
    matched = (d_s <= max_snap_dist) & (d_t <= max_snap_dist)
    diffs = np.ones(len(L1))                  # unmatched / unreachable -> zero credit
    if matched.any():
        L2 = _pair_lengths(_csr(G_dst, dst_nodes), i_s[matched], i_t[matched])
        l1 = L1[matched]
        fin = np.isfinite(L2)
        d = np.ones(len(l1))
        d[fin] = np.minimum(1.0, np.abs(l1[fin] - L2[fin]) / np.maximum(l1[fin], 1e-6))
        diffs[matched] = d
    return 1.0 - float(np.mean(diffs))


def apls(G_pred: nx.Graph, G_true: nx.Graph, sample: int = 500, seed: int = 0,
         max_snap_dist: float = 8.0) -> float:
    """Symmetric APLS in [0, 1], higher is better. Both graphs empty -> 1.0;
    exactly one empty -> 0.0."""
    n_p, n_t = G_pred.number_of_nodes(), G_true.number_of_nodes()
    if n_p == 0 and n_t == 0:
        return 1.0
    if n_p == 0 or n_t == 0:
        return 0.0
    _ensure_edge_lengths(G_pred); _ensure_edge_lengths(G_true)
    rng = np.random.default_rng(seed)
    a = _one_way(G_true, G_pred, sample, rng, max_snap_dist)
    b = _one_way(G_pred, G_true, sample, rng, max_snap_dist)
    return 0.5 * (a + b)
