import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from src.routing.graph_manager import GraphManager

H5_PATH   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "simulation_data.h5")
CONST_CFG = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "constellation.yaml")


def test_get_snapshot_graph_smoke():
    """Smoke test: HDF5 loads and get_snapshot_graph returns non-empty dicts."""
    gm = GraphManager(H5_PATH, CONST_CFG)
    neighbors, costs = gm.get_snapshot_graph(0)

    assert isinstance(neighbors, dict), "neighbors must be a dict"
    assert len(neighbors) > 0, "neighbors must not be empty"
    assert isinstance(costs, dict), "costs must be a dict"
    assert len(costs) > 0, "costs must not be empty"
    for (i, j), d in costs.items():
        assert isinstance(i, int) and isinstance(j, int)
        assert d >= 0.0
    print("  PASS test_get_snapshot_graph_smoke")


def test_snapshot_graph_node_count():
    """Neighbor dict must have exactly num_nodes entries."""
    gm = GraphManager(H5_PATH, CONST_CFG)
    neighbors, _ = gm.get_snapshot_graph(0)
    assert len(neighbors) == gm.num_nodes, (
        f"expected {gm.num_nodes} nodes, got {len(neighbors)}"
    )
    print("  PASS test_snapshot_graph_node_count")


def test_switch_matrix_single_new_link():
    """NS: H_prev all-zeros + H_curr with one 1 → NS has exactly one 1."""
    N      = 4
    H_prev = np.zeros((N, N), dtype=int)
    H_curr = np.zeros((N, N), dtype=int)
    H_curr[0, 1] = 1

    NS = ((H_curr == 1) & (H_prev == 0)).astype(int)

    assert int(np.sum(NS)) == 1, f"expected 1 new link, got {np.sum(NS)}"
    assert NS[0, 1] == 1, "new link should be at [0,1]"
    print("  PASS test_switch_matrix_single_new_link")


def test_switch_matrix_existing_link_not_counted():
    """NS: a link present in both H_prev and H_curr should not appear in NS."""
    N      = 4
    H_prev = np.zeros((N, N), dtype=int)
    H_curr = np.zeros((N, N), dtype=int)
    H_prev[0, 1] = 1
    H_curr[0, 1] = 1

    NS = ((H_curr == 1) & (H_prev == 0)).astype(int)

    assert int(np.sum(NS)) == 0, "unchanged link must not appear in NS"
    print("  PASS test_switch_matrix_existing_link_not_counted")


def test_check_degree_constraints_no_violations():
    """Paths with low degree should produce zero violations."""
    gm    = GraphManager(H5_PATH, CONST_CFG)
    paths = [[0, 1], [2, 3]]
    assert gm.check_degree_constraints(paths, gm.num_nodes) == 0
    print("  PASS test_check_degree_constraints_no_violations")


def test_check_degree_constraints_with_violations():
    """5 paths all through node 0 exceed o_i_max=4 → at least 1 violation."""
    gm = GraphManager(H5_PATH, CONST_CFG)
    # node 0 has out-degree 5, exceeding o_i_max=4
    paths = [[0, k] for k in range(1, 6)]
    violations = gm.check_degree_constraints(paths, gm.num_nodes)
    assert violations >= 1, f"expected ≥1 violation, got {violations}"
    print("  PASS test_check_degree_constraints_with_violations")


if __name__ == "__main__":
    test_get_snapshot_graph_smoke()
    test_snapshot_graph_node_count()
    test_switch_matrix_single_new_link()
    test_switch_matrix_existing_link_not_counted()
    test_check_degree_constraints_no_violations()
    test_check_degree_constraints_with_violations()
    print("\nAll test_graph_manager tests passed.")
