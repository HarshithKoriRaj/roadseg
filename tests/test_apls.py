import networkx as nx, numpy as np, pytest
from roadseg.apls import apls, graph_from_mask


def _line_graph(n=10, dx=0.0):
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, y=0.0, x=float(i * 10) + dx)
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
