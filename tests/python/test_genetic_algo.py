import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import random
import numpy as np
from src.optimization.genetic_algo import improved_a_star, GA_IAS

# 4 node dummy graph
NB = {0: [1, 2], 1: [3], 2: [3], 3: []}
CB = {(0, 1): 1.0, (0, 2): 1.0, (1, 3): 1.0, (2, 3): 1.0}
DB = np.array(
    [[0, 1, 1, 2],
     [1, 0, 2, 1],
     [1, 2, 0, 1],
     [2, 1, 1, 0]],
    dtype=float,
)


class _FakeGM:
    num_nodes = 4
    o_i_max   = 4
    l_i_max   = 4

    def check_degree_constraints(self, paths, n):
        out = [0] * n
        inp = [0] * n
        for path in paths:
            for k in range(len(path) - 1):
                out[path[k]] += 1
                inp[path[k + 1]] += 1
        return sum(
            1 for v in range(n)
            if out[v] > self.o_i_max or inp[v] > self.l_i_max
        )



def test_improved_a_star_returns_valid_path():
    """Returns a connected path from start to end."""
    path = improved_a_star(0, 3, NB, CB, DB)
    assert path is not None, "expected a path, got None"
    assert path[0] == 0, f"path must start at 0, got {path[0]}"
    assert path[-1] == 3, f"path must end at 3, got {path[-1]}"
    for i in range(len(path) - 1):
        assert path[i + 1] in NB[path[i]], (
            f"invalid hop {path[i]} → {path[i+1]}"
        )
    print("  PASS test_improved_a_star_returns_valid_path")


def test_improved_a_star_disconnected_returns_none():
    """No route exists → returns None."""
    nb = {0: [1], 1: [], 2: [3], 3: []}
    cb = {(0, 1): 1.0}
    D  = np.ones((4, 4))
    np.fill_diagonal(D, 0.0)
    assert improved_a_star(0, 3, nb, cb, D) is None
    print("  PASS test_improved_a_star_disconnected_returns_none")


def test_improved_a_star_trivial_start_equals_end():
    """start == end → single-node list, no A* needed."""
    path = improved_a_star(2, 2, NB, CB, DB)
    assert path == [2], f"expected [2], got {path}"
    print("  PASS test_improved_a_star_trivial_start_equals_end")


def test_improved_a_star_produces_diversity():
    """Stochastic selection should produce at least two distinct paths."""
    random.seed(0)
    seen = set()
    for _ in range(40):
        p = improved_a_star(0, 3, NB, CB, DB)
        assert p is not None
        seen.add(tuple(p))
    assert len(seen) > 1, f"expected path diversity, only got: {seen}"
    print("  PASS test_improved_a_star_produces_diversity")


def test_ga_ias_returns_correct_types():
    """GA_IAS output types: list, ndarray, float."""
    random.seed(42)
    tasks  = [(0, 3, 1.0), (0, 3, 0.8)]
    H_prev = np.zeros((4, 4), dtype=int)

    best_ind, H_opt, best_fit = GA_IAS(
        tasks          = tasks,
        neighbors      = NB,
        costs          = CB,
        D_cost         = DB,
        B_snapshot     = np.zeros((4, 4)),
        graph_manager  = _FakeGM(),
        H_prev         = H_prev,
        weights        = {"w1": 0.8, "w2": 0.1, "w3": 0.1},
        p_coefficients = {"p1": 1.0, "p2": 1.0, "p3": 1.0},
        population_size = 10,
        max_iterations  = 5,
        eta_m = 0.1, eta_c = 0.5, eta_g = 0.5, M = 10000.0,
    )

    assert isinstance(best_ind, list),        "best_individual must be a list"
    assert isinstance(H_opt, np.ndarray),     "H_opt must be ndarray"
    assert H_opt.shape == (4, 4),             f"H_opt shape wrong: {H_opt.shape}"
    assert isinstance(best_fit, float),       "best_fitness must be a float"
    print("  PASS test_ga_ias_returns_correct_types")


def test_ga_ias_smoke_five_tasks():
    """GA-AS runs 5 iterations on a 5-task subset and returns non-None paths."""
    random.seed(7)
    tasks  = [(0, 3, 1.0)] * 5
    H_prev = np.zeros((4, 4), dtype=int)

    best_ind, H_opt, best_fit = GA_IAS(
        tasks          = tasks,
        neighbors      = NB,
        costs          = CB,
        D_cost         = DB,
        B_snapshot     = np.zeros((4, 4)),
        graph_manager  = _FakeGM(),
        H_prev         = H_prev,
        weights        = {"w1": 0.8, "w2": 0.1, "w3": 0.1},
        p_coefficients = {"p1": 1.0, "p2": 1.0, "p3": 1.0},
        population_size = 10,
        max_iterations  = 5,
        eta_m = 0.1, eta_c = 0.5, eta_g = 0.5, M = 10000.0,
    )

    assert best_ind is not None, "best_individual must not be None"
    assert len(best_ind) == len(tasks), "one path entry per task"
    for path in best_ind:
        if path is not None:
            assert path[0] == 0 and path[-1] == 3
    print("  PASS test_ga_ias_smoke_five_tasks")


def test_ga_ias_h_opt_reflects_paths():
    """H_opt must have 1s on every edge used in best_individual."""
    random.seed(0)
    tasks  = [(0, 3, 1.0)]
    H_prev = np.zeros((4, 4), dtype=int)

    best_ind, H_opt, _ = GA_IAS(
        tasks          = tasks,
        neighbors      = NB,
        costs          = CB,
        D_cost         = DB,
        B_snapshot     = np.zeros((4, 4)),
        graph_manager  = _FakeGM(),
        H_prev         = H_prev,
        weights        = {"w1": 0.8, "w2": 0.1, "w3": 0.1},
        p_coefficients = {"p1": 1.0, "p2": 1.0, "p3": 1.0},
        population_size = 10,
        max_iterations  = 5,
        eta_m = 0.1, eta_c = 0.5, eta_g = 0.5, M = 10000.0,
    )

    for path in best_ind:
        if path is not None:
            for k in range(len(path) - 1):
                assert H_opt[path[k], path[k + 1]] == 1, (
                    f"H_opt missing edge {path[k]}→{path[k+1]}"
                )
    print("  PASS test_ga_ias_h_opt_reflects_paths")


if __name__ == "__main__":
    test_improved_a_star_returns_valid_path()
    test_improved_a_star_disconnected_returns_none()
    test_improved_a_star_trivial_start_equals_end()
    test_improved_a_star_produces_diversity()
    test_ga_ias_returns_correct_types()
    test_ga_ias_smoke_five_tasks()
    test_ga_ias_h_opt_reflects_paths()
    print("\nAll test_genetic_algo tests passed.")
