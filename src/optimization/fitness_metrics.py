import numpy as np


def compute_psi(path, B_snapshot):
    """
    Eq. 22, the probability of the set of communication tasks in the cth topology represented by the static directed graph G(c) being executed
    """
    psi = 1.0
    for hop in range(len(path) - 1):
        i, j = path[hop], path[hop + 1]
        psi *= (1.0 - B_snapshot[i, j])
    return psi


def compute_f1(paths, B_snapshot, priorities):
    """
    Eq. 21, revenue of tasks as sum of the priority * the aforementioned probability psi, for each task path 
    in layman terms, f1 is a measure of how much data the network successfully transfers, with higher priority tasks contributing more to the score. 
    """
    f1 = 0.0
    for path, priority in zip(paths, priorities):
        psi_k = compute_psi(path, B_snapshot)
        f1 += psi_k * priority
    return f1


def compute_f2(NS_matrix):
    """
    Eq. 23, number of all new links in the graph G(c), as found from NS_matrix created from compute_switch_matrix() in graph_manager.py
    in layman terms, f2 is a measure of how many new links have been added to the network. More new links being established is bad, we want to minize f2. 
    """
    return float(np.sum(NS_matrix))


def compute_f3(paths, D_snapshot):
    """
    Eq. 24, total cost of graph G(c) 
    in layman terms, f3 measures the total propagation distance (propagation delay) of the data. We want to minimize f3, shorter paths are better. 
    """
    f3 = 0.0
    for path in paths:
        for hop in range(len(path) - 1):
            i, j = path[hop], path[hop + 1]
            f3 += D_snapshot[i, j]
    return f3


def compute_objective(f1, f2, f3, weights, p_coefficients):
    """
    Eq. 25, the objective function which we aim to maximize in GA-A* algorithm.
    Total equation which combines the three metrics, and we optimize for this individual objective instead of all 3 seperately. 
    """

    if isinstance(weights, dict):
        w1, w2, w3 = weights["w1"], weights["w2"], weights["w3"]
    else:
        w1, w2, w3 = weights[0], weights[1], weights[2]

    if isinstance(p_coefficients, dict):
        p1, p2, p3 = p_coefficients["p1"], p_coefficients["p2"], p_coefficients["p3"]
    else:
        p1, p2, p3 = p_coefficients[0], p_coefficients[1], p_coefficients[2]

    return (w1 * p1 * f1) - (w2 * p2 * f2) - (w3 * p3 * f3)


def compute_fitness(f, n_violations, M):
    """
    Eq. 26, fitness value 
    f but we make f really bad if violations are present. 
    """
    return f - n_violations * M
