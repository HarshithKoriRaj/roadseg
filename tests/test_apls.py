import math

import networkx as nx, numpy as np, pytest
from roadseg.apls import apls, graph_from_mask


def _line_graph(n=10, dx=0.0, dy=0.0):
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, y=dy, x=float(i * 10) + dx)
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    return G


def test_identical_graphs_score_one():
    G = _line_graph()
    assert apls(G, G, sample=50) == pytest.approx(1.0, abs=1e-6)


def test_small_offset_still_scores_high():
    assert apls(_line_graph(dx=2.0), _line_graph(), sample=50, max_snap_dist=8.0) > 0.9


def test_far_graph_scores_zero_not_high():
    """THE defect in the ported implementation: without max_snap_dist a distant
    prediction still snapped to a nearest node and earned near-full credit."""
    far = _line_graph(dx=10_000.0)
    assert apls(far, _line_graph(), sample=50, max_snap_dist=8.0) < 0.05


def test_threshold_is_what_zeroes_a_shifted_copy():
    """Same topology shifted 100 px: pinned to max_snap_dist. Without the
    threshold every node snaps to its counterpart and scores ~1."""
    shifted, T = _line_graph(dy=100.0), _line_graph()
    assert apls(shifted, T, sample=50, max_snap_dist=8.0) < 0.05
    assert apls(shifted, T, sample=50, max_snap_dist=1e12) > 0.9


def test_both_empty_scores_one_and_one_empty_scores_zero():
    assert apls(nx.Graph(), nx.Graph()) == 1.0
    assert apls(_line_graph(), nx.Graph()) == 0.0


def test_deterministic_for_a_seed():
    a, b = _line_graph(dx=3.0), _line_graph()
    assert apls(a, b, sample=30, seed=5) == apls(a, b, sample=30, seed=5)


def test_graph_from_mask_empty_and_thick_road():
    assert graph_from_mask(np.zeros((32, 32), bool)).number_of_nodes() == 0
    m = np.zeros((64, 64), bool); m[30:35, 5:60] = True
    assert nx.number_connected_components(graph_from_mask(m)) == 1


def _brute_one_way(Gs, Gd, sample, rng, snap):
    nodes = list(Gs.nodes())
    dn = list(Gd.nodes())
    dp = np.array([[Gd.nodes[n]["y"], Gd.nodes[n]["x"]] for n in dn])
    diffs = []
    for _ in range(sample):
        i, j = rng.choice(len(nodes), size=2, replace=False)
        s, t = nodes[int(i)], nodes[int(j)]
        try:
            L1 = nx.shortest_path_length(Gs, s, t, weight="length")
        except nx.NetworkXNoPath:
            continue
        if L1 <= 0:
            continue
        q = [np.array([Gs.nodes[n]["y"], Gs.nodes[n]["x"]]) for n in (s, t)]
        dd = [np.hypot(*(dp - p).T) for p in q]
        a, b = int(np.argmin(dd[0])), int(np.argmin(dd[1]))
        if dd[0][a] > snap or dd[1][b] > snap:
            diffs.append(1.0); continue
        try:
            L2 = nx.shortest_path_length(Gd, dn[a], dn[b], weight="length")
            diffs.append(min(1.0, abs(L1 - L2) / max(L1, 1e-6)))
        except nx.NetworkXNoPath:
            diffs.append(1.0)
    return 0.0 if not diffs else 1.0 - float(np.mean(diffs))


def test_batched_dijkstra_matches_bruteforce_reference():
    rs = np.random.default_rng(1)

    def grid(jitter, drop):
        G = nx.Graph()
        for y in range(7):
            for x in range(7):
                dy, dx = rs.uniform(-jitter, jitter, 2) if jitter else (0, 0)
                G.add_node(y * 7 + x, y=y * 6.0 + dy, x=x * 6.0 + dx)
        for y in range(7):
            for x in range(7):
                if x < 6: G.add_edge(y * 7 + x, y * 7 + x + 1)
                if y < 6: G.add_edge(y * 7 + x, (y + 1) * 7 + x)
        for e in list(G.edges())[::drop] if drop else []:
            G.remove_edge(*e)
        for u, v, d in G.edges(data=True):
            d["length"] = math.hypot(G.nodes[u]["y"] - G.nodes[v]["y"],
                                     G.nodes[u]["x"] - G.nodes[v]["x"])
        return G

    T, P = grid(0, 0), grid(2.0, 9)
    got = apls(P, T, sample=80, seed=3, max_snap_dist=3.0)
    rng = np.random.default_rng(3)
    ref = 0.5 * (_brute_one_way(T, P, 80, rng, 3.0) + _brute_one_way(P, T, 80, rng, 3.0))
    assert 0.0 < got < 1.0
    assert got == pytest.approx(ref, abs=1e-9)


def test_empty_prediction_scores_zero():
    assert apls(nx.Graph(), _line_graph(), sample=50) == 0.0


def test_graph_from_mask_recovers_a_line():
    m = np.zeros((64, 64), bool); m[32, 8:56] = True
    G = graph_from_mask(m)
    assert G.number_of_nodes() >= 2
    assert nx.number_connected_components(G) == 1


def test_graph_from_mask_off_lattice_line_is_connected():
    m = np.zeros((64, 64), bool); m[33, 8:56] = True
    assert nx.number_connected_components(graph_from_mask(m)) == 1


def test_graph_from_mask_diagonal_is_not_fragmented():
    m = np.zeros((64, 64), bool)
    for i in range(10, 54):
        m[i, i] = True
    m2 = np.zeros((64, 64), bool)
    for i in range(10, 54):
        m2[i, i + 3] = True            # off-lattice diagonal
    for mm in (m, m2):
        assert nx.number_connected_components(graph_from_mask(mm)) == 1


def test_graph_from_mask_plus_intersection_is_one_component():
    m = np.zeros((64, 64), bool); m[33, 8:56] = True; m[8:56, 35] = True
    G = graph_from_mask(m)
    assert nx.number_connected_components(G) == 1
    ys = [d["y"] for _, d in G.nodes(data=True)]
    assert min(ys) < 12 and max(ys) > 50      # both arms present


def test_graph_from_mask_separate_roads_stay_separate():
    far = np.zeros((64, 64), bool); far[10, 8:56] = True; far[50, 8:56] = True
    assert nx.number_connected_components(graph_from_mask(far)) == 2
    close = np.zeros((64, 64), bool); close[30, 8:56] = True; close[34, 8:56] = True
    assert nx.number_connected_components(graph_from_mask(close)) == 2


def test_graph_from_mask_edge_lengths_are_path_lengths():
    m = np.zeros((64, 64), bool); m[33, 8:56] = True
    G = graph_from_mask(m)
    total = sum(d["length"] for _, _, d in G.edges(data=True))
    assert total == pytest.approx(47.0, abs=1.0)


def test_runs_on_a_large_graph_and_uses_a_kdtree():
    """Completion + sanity on a big graph, and a structural guard against the
    O(N^2) nearest-node scan returning."""
    import inspect, roadseg.apls as mod
    m = np.zeros((512, 512), bool)
    m[::32, :] = True; m[:, ::32] = True
    G = graph_from_mask(m)
    assert G.number_of_nodes() > 1000
    score = apls(G, G, sample=40)
    assert score == pytest.approx(1.0, abs=1e-6)
    src = inspect.getsource(mod)
    assert "cKDTree" in src and "_nearest_node" not in src
