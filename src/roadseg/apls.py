"""APLS (Average Path Length Similarity) for rasterised road predictions.

Ported from sam2_road_dac with two corrections:

1. Nearest-node lookup uses a cKDTree. The original scanned every node per
   query, making the metric O(N^2) on graphs with thousands of nodes.
2. A `max_snap_dist` threshold is enforced. The original snapped to the nearest
   node at any distance, so a prediction whose nearest node was far away still
   received credit; the standard metric scores an unmatched node 0. Without
   this the reported APLS is systematically inflated.
"""
from __future__ import annotations

import heapq
import math

import networkx as nx
import numpy as np
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


def _shortest_len(G: nx.Graph, s, t) -> float:
    try:
        return float(nx.shortest_path_length(G, s, t, weight="length"))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")


def _one_way(G_src: nx.Graph, G_dst: nx.Graph, sample: int, rng, max_snap_dist: float) -> float:
    src_nodes = list(G_src.nodes())
    if len(src_nodes) < 2 or G_dst.number_of_nodes() == 0:
        return 0.0
    dst_pts, dst_nodes = _pos_array(G_dst)
    tree = cKDTree(dst_pts)
    diffs = []
    for _ in range(sample):
        i, j = rng.choice(len(src_nodes), size=2, replace=False)
        s, t = src_nodes[int(i)], src_nodes[int(j)]
        L1 = _shortest_len(G_src, s, t)
        if not math.isfinite(L1) or L1 <= 0:
            continue
        sq = [[G_src.nodes[s]["y"], G_src.nodes[s]["x"]],
              [G_src.nodes[t]["y"], G_src.nodes[t]["x"]]]
        dists, idxs = tree.query(np.array(sq), k=1)
        if dists[0] > max_snap_dist or dists[1] > max_snap_dist:
            diffs.append(1.0)          # unmatched -> zero credit
            continue
        L2 = _shortest_len(G_dst, dst_nodes[int(idxs[0])], dst_nodes[int(idxs[1])])
        diffs.append(1.0 if not math.isfinite(L2) else min(1.0, abs(L1 - L2) / max(L1, 1e-6)))
    return 0.0 if not diffs else 1.0 - float(np.mean(diffs))


def apls(G_pred: nx.Graph, G_true: nx.Graph, sample: int = 500, seed: int = 0,
         max_snap_dist: float = 8.0) -> float:
    """Symmetric APLS in [0, 1], higher is better."""
    if G_pred.number_of_nodes() == 0 or G_true.number_of_nodes() == 0:
        return 0.0
    _ensure_edge_lengths(G_pred); _ensure_edge_lengths(G_true)
    rng = np.random.default_rng(seed)
    a = _one_way(G_true, G_pred, sample, rng, max_snap_dist)
    b = _one_way(G_pred, G_true, sample, rng, max_snap_dist)
    return 0.5 * (a + b)
