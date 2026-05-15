import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from src.optimization.fitness_metrics import (
    compute_psi,
    compute_f1,
    compute_f2,
    compute_f3,
    compute_objective,
    compute_fitness,
)

def test_psi_empty_path():
    """Empty path"""
    B = np.zeros((4, 4))
    assert compute_psi([], B) == 1.0
    print("  PASS test_psi_empty_path")


def test_psi_single_node_path():
    """Single-node path"""
    B = np.zeros((4, 4))
    assert compute_psi([0], B) == 1.0
    print("  PASS test_psi_single_node_path")


def test_psi_single_link_half_failure():
    """One hop with b=0.5"""
    B = np.zeros((4, 4))
    B[0, 1] = 0.5
    result = compute_psi([0, 1], B)
    assert abs(result - 0.5) < 1e-9, f"expected 0.5, got {result}"
    print("  PASS test_psi_single_link_half_failure")


def test_psi_two_hops():
    """Two hops, each b=0.5"""
    B = np.zeros((4, 4))
    B[0, 1] = 0.5
    B[1, 2] = 0.5
    result = compute_psi([0, 1, 2], B)
    assert abs(result - 0.25) < 1e-9, f"expected 0.25, got {result}"
    print("  PASS test_psi_two_hops")


def test_psi_perfect_links():
    """All b=0, psi = 1.0 regardless of path length."""
    B = np.zeros((5, 5))
    result = compute_psi([0, 1, 2, 3, 4], B)
    assert abs(result - 1.0) < 1e-9
    print("  PASS test_psi_perfect_links")


def test_f1_single_task_perfect_link():
    """f1 = priority when all links are perfect (b=0)."""
    B = np.zeros((4, 4))
    paths      = [[0, 1]]
    priorities = [2.0]
    result = compute_f1(paths, B, priorities)
    assert abs(result - 2.0) < 1e-9, f"expected 2.0, got {result}"
    print("  PASS test_f1_single_task_perfect_link")


def test_f1_two_tasks():
    """f1 sums over all tasks: priority_k * psi_k."""
    B = np.zeros((4, 4))
    B[2, 3] = 0.5                       # second task path has psi=0.5
    paths      = [[0, 1], [2, 3]]
    priorities = [1.0, 2.0]
    # f1 = 1.0*1.0 + 2.0*0.5 = 2.0
    result = compute_f1(paths, B, priorities)
    assert abs(result - 2.0) < 1e-9, f"expected 2.0, got {result}"
    print("  PASS test_f1_two_tasks")


def test_f2_known_value():
    """f2 = sum of all entries in NS matrix."""
    NS = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=int)
    assert compute_f2(NS) == 2.0
    print("  PASS test_f2_known_value")


def test_f2_zero_matrix():
    """All-zeros NS → f2 = 0."""
    NS = np.zeros((4, 4), dtype=int)
    assert compute_f2(NS) == 0.0
    print("  PASS test_f2_zero_matrix")


def test_f3_single_path():
    """f3 = sum of edge distances along the path."""
    D = np.zeros((4, 4))
    D[0, 1] = 3.0
    D[1, 2] = 4.0
    result = compute_f3([[0, 1, 2]], D)
    assert abs(result - 7.0) < 1e-9, f"expected 7.0, got {result}"
    print("  PASS test_f3_single_path")


def test_f3_two_paths():
    """f3 sums over multiple paths."""
    D = np.zeros((4, 4))
    D[0, 1] = 1.0
    D[2, 3] = 2.0
    result = compute_f3([[0, 1], [2, 3]], D)
    assert abs(result - 3.0) < 1e-9, f"expected 3.0, got {result}"
    print("  PASS test_f3_two_paths")


def test_compute_objective_formula():
    """Objective = w1*p1*f1 - w2*p2*f2 - w3*p3*f3."""
    weights = {"w1": 1.0, "w2": 1.0, "w3": 1.0}
    p       = {"p1": 1.0, "p2": 1.0, "p3": 1.0}
    result  = compute_objective(f1=5.0, f2=2.0, f3=1.0, weights=weights, p_coefficients=p)
    assert abs(result - 2.0) < 1e-9, f"expected 2.0, got {result}"  # 5 - 2 - 1
    print("  PASS test_compute_objective_formula")


def test_compute_objective_list_weights():
    """compute_objective also accepts list-style weights."""
    result = compute_objective(
        f1=10.0, f2=0.0, f3=0.0,
        weights=[0.8, 0.1, 0.1],
        p_coefficients=[1.0, 1.0, 1.0],
    )
    assert abs(result - 8.0) < 1e-9
    print("  PASS test_compute_objective_list_weights")


def test_fitness_no_violations():
    """With zero violations, fitness equals objective."""
    result = compute_fitness(f=3.5, n_violations=0, M=10000.0)
    assert abs(result - 3.5) < 1e-9
    print("  PASS test_fitness_no_violations")


def test_fitness_penalty_one_violation():
    """One violation subtracts M from the objective."""
    result = compute_fitness(f=0.0, n_violations=1, M=10000.0)
    assert abs(result - (-10000.0)) < 1e-9
    print("  PASS test_fitness_penalty_one_violation")


def test_fitness_penalty_multiple_violations():
    """n violations subtract n*M from the objective."""
    result = compute_fitness(f=5.0, n_violations=3, M=1000.0)
    assert abs(result - (5.0 - 3000.0)) < 1e-9
    print("  PASS test_fitness_penalty_multiple_violations")


if __name__ == "__main__":
    test_psi_empty_path()
    test_psi_single_node_path()
    test_psi_single_link_half_failure()
    test_psi_two_hops()
    test_psi_perfect_links()
    test_f1_single_task_perfect_link()
    test_f1_two_tasks()
    test_f2_known_value()
    test_f2_zero_matrix()
    test_f3_single_path()
    test_f3_two_paths()
    test_compute_objective_formula()
    test_compute_objective_list_weights()
    test_fitness_no_violations()
    test_fitness_penalty_one_violation()
    test_fitness_penalty_multiple_violations()
    print("\nAll test_fitness_metrics tests passed.")
