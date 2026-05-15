import numpy as np
import networkx as nx
import random
from .fitness_metrics import compute_f1, compute_f3, compute_objective, compute_fitness


def selection_probability(nodes: list, h_values: dict, epsilon: float = 1e-6) -> dict:
    """
    Eq. 27, selection probability for roulette-wheel selection in improved A* algo 
    """

    # compute numerator 1/h(v_si) for all nodes
    selection_probabilities = {}
    for v in nodes:
        numerator = 1.0/(h_values[v]+epsilon)
        selection_probabilities[v] = numerator

    # denomator of selection probability equation 
    total = sum(selection_probabilities.values())

    # update dict with selection_probability value for each node 
    for node in selection_probabilities: 
        selection_probabilities[node] = selection_probabilities[node]/total
    
    return selection_probabilities


def cumulative_probability(nodes: list, probs: dict) -> list:
    """
    Eq. 28, cumulative probability of any node 
    """

    cum_probs = []
    cumsum = 0.0 # lol 
    for v in nodes:
        cumsum += probs[v]
        cum_probs.append(cumsum) 

    return cum_probs


def _roulette_select(nodes: list, cum_probs: list, z_s: float) -> int:
    """
    Eq. 32, next path node selection based off of random variable z 
    """
    for node, q in zip(nodes, cum_probs):
        if z_s <= q:
            return node
    return nodes[-1]


def improved_a_star(
    start_node: int,
    end_node: int,
    neighbors: dict,
    costs: dict,
    D_cost,
    alpha: float = 1.0,
    beta: float = 1.0,
    epsilon: float = 1e-6,
):
    """
    Algorithm 2 from the paper
    """

    if start_node == end_node:
        return [start_node]

    
    open_list = [start_node]            # open_list tracks nodes we have discovered but not yet evaluated
    closed_set = set()                  # tracks nodes we have visited 
    h1 = {start_node: 0.0}              
    parent = {start_node: None}

    while open_list:
    
        h_values = {}
        for node in open_list:
            g_score = h1[node]                              # cost from start to current node 
            h_score = float(D_cost[node, end_node])         # heuristic distance from current node to end destination 
            h_values[node] = g_score + h_score 

        z_s = random.uniform(0.0, 1.0)
        probs = selection_probability(open_list, h_values, epsilon)
        cum_probs = cumulative_probability(open_list, probs)
        current = _roulette_select(open_list, cum_probs, z_s)

        # there was a typo in the paper, fairly certain they want to add a to close list and remvoe it from open
        open_list.remove(current)
        closed_set.add(current)

        if current == end_node:
            break

        for child in neighbors.get(current, []):
            if child in closed_set:
                continue

            # h1 (actual path cost) for child through eq. 30 
            new_h1 = h1[current] + costs.get((current, child), float("inf"))

            if child in open_list:
                if new_h1 < h1[child]:
                    h1[child] = new_h1
                    parent[child] = current
            else:
                h1[child] = new_h1
                parent[child] = current
                open_list.append(child)

    # No feasible path: end_node was never added to the Open List
    if end_node not in parent:
        return None

    # Reconstruct path
    path = []
    node = end_node
    while node is not None:
        path.append(node)
        node = parent.get(node)
    path.reverse()

    return path





























def _build_H(individual: list, n_nodes: int) -> np.ndarray:
    """
    individual is a list of paths per task. So each individual carries up to n paths for task 1, task 2, ... task n. 
    Build optimal connection matrix from an individual's path set.  
    -> this serves as the active routing topology, looks very similar to adjacency matrix
    """
    H = np.zeros((n_nodes, n_nodes), dtype=int)
    for path in individual:
        if path is not None:
            for k in range(len(path) - 1):
                H[path[k], path[k + 1]] = 1
    return H


def _evaluate_individual(
    individual: list,
    tasks,
    B_snapshot,
    D_cost,
    H_prev: np.ndarray,
    graph_manager,
    weights,
    p_coefficients,
    M: float,
) -> float:
    """
    provides a score for each individual (list of paths) in the population. As per the GA-AS algorithm, this is steps 3-5 
    1)  calculate the objective function value f
    2)  count the number of constraint violations n for per individual
    3)  calculate fitness value Faccording to Eq. 26
    """

    n_nodes = graph_manager.num_nodes
    
    priorities = []                             # extracts task priorities, randomly generated from graph_manager 
    for t in tasks: 
        task_priority = float(t[2])
        priorities.append(task_priority)

    valid_paths = []
    valid_priorities = []

    # attach valid paths and their priorities to seperate lists 
    for k in range(len(individual)):
        path = individual[k]
        if path is not None:
            valid_paths.append(path)
            valid_priorities.append(priorities[k])

    # calculate f1, total task revenue
    if len(valid_paths) > 0:
        f1 = compute_f1(valid_paths, B_snapshot, valid_priorities)
    else:
        f1 = 0.0

    # calculate f2, network switching cost 
    H_curr = _build_H(individual, n_nodes)
    NS = ((H_curr == 1) & (H_prev == 0)).astype(int)    # binary matrix where 1 indicates a new link switch from previous snapshot to current snapshot
    f2 = float(np.sum(NS))

    # calculate f3, total path cost
    if len(valid_paths) > 0:
        f3 = compute_f3(valid_paths, D_cost)
    else:
        f3 = 0.0

    # compute objective function f 
    f = compute_objective(f1, f2, f3, weights, p_coefficients)

    # Check for physical constraint violations
    n_violations = graph_manager.check_degree_constraints(valid_paths, n_nodes)
    for p in individual:
        if p is None:
            n_violations += 1

    # final fitness
    final_fitness = compute_fitness(f, n_violations, M)

    return final_fitness


def _tournament_select(population: list, fitnesses: list, k: int = 3) -> list:
    """
    randomly select k individuals from the population, then return the one with the highest fitness.
    """
    population_size = len(population)

    # draft k individuals for tournament selection
    sample_size = min(k, population_size)
    contender_indices = random.sample(range(population_size), sample_size)

    # select the individual with the highest fitness among the contenders
    winning_index = contender_indices[0]
    highest_fitness = fitnesses[winning_index]
    
    for idx in contender_indices:
        if fitnesses[idx] > highest_fitness:
            highest_fitness = fitnesses[idx]
            winning_index = idx
    
    winning_individual = population[winning_index]

    # clean copy of winning individuals paths 
    copied_paths = []
    for path in winning_individual:
        if path is not None:
            copied_paths.append(path[:]) 
        else:
            copied_paths.append(None)

    return copied_paths


def _crossover(parent1: list, parent2: list, eta_c: float):
    """
    single-point crossover: mechanism for heredity in line 6 of algorithm 1 
    1) take two high-performing parent individuals and slices them at random spots to mix and match their paths 
    """

    # make a copy of the parent individuals 
    def _copy(ind):
        cloned_ind = []
        for path in ind:
            if path is not None:
                cloned_ind.append(path[:])
            else:
                cloned_ind.append(None)
        return cloned_ind

    if random.random() > eta_c or len(parent1) < 2:
        return _copy(parent1), _copy(parent2)  # identical clones of parents

    total_tasks = len(parent1)
    split_point = random.randint(1, total_tasks - 1)

    parent1_left_side  = parent1[:split_point]
    parent1_right_side = parent1[split_point:]

    parent2_left_side  = parent2[:split_point]
    parent2_right_side = parent2[split_point:]

    child1_paths = parent1_left_side + parent2_right_side
    child2_paths = parent2_left_side + parent1_right_side

    return _copy(child1_paths), _copy(child2_paths)
    

def _mutate(
    individual: list,
    tasks,
    neighbors: dict,
    costs: dict,
    D_cost,
    eta_m: float,
    alpha: float,
    beta: float,
    epsilon: float,
) -> list:

    """
    Creates random mutations: calling Algorithm 2 (A*) and regenerating a path for a specific task inside that individual 
    """

    # clone paths 
    mutated_individual = []
    for path in individual:
        if path is not None:
            mutated_individual.append(path[:])
        else:
            mutated_individual.append(None)

            
    for k in range(len(tasks)):
        current_task = tasks[k]

        # randomly mutate a path based on mutation_probability 
        if random.random() < eta_m:
            source_node = int(current_task[0])
            destination_node = int(current_task[1])

            alt_path = improved_a_star(
                source_node, 
                destination_node,
                neighbors, 
                costs, 
                D_cost,
                alpha, 
                beta, 
                epsilon
            )

            mutated_individual[k] = alt_path

    return mutated_individual


def GA_IAS(
    tasks,
    neighbors: dict,
    costs: dict,
    D_cost,
    B_snapshot,
    graph_manager,
    H_prev: np.ndarray,
    weights,
    p_coefficients,
    population_size: int,
    max_iterations: int,
    eta_m: float,
    eta_c: float,
    eta_g: float,
    M: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    epsilon: float = 1e-6,
    positions_c=None,
):
    """
    Algorithm 1: Improved Genetic Algorithm Based on A* (GA-AS).

    Inputs: 
        tasks            : (K, 3) array of (src, dst, priority) per task.
        neighbors        : adjacency list from get_snapshot_graph(c).
        costs            : edge distances (km) from get_snapshot_graph(c).
        D_cost           : (N, N) pairwise distance matrix for snapshot c.
        B_snapshot       : (N, N) link failure probabilities for snapshot c.
        graph_manager    : provides num_nodes and check_degree_constraints().
        H_prev           : (N, N) connection matrix from previous snapshot.
        weights          : [w1, w2, w3] objective weights (Eq. 25).
        p_coefficients   : [p1, p2, p3] scaling coefficients (Eq. 25).
        population_size  : number of individuals.
        max_iterations   : number of GA generations.
        eta_m            : per-gene mutation probability.
        eta_c            : crossover probability.
        eta_g            : elitist selection retention fraction.
        M                : constraint violation penalty constant (Eq. 26).
        alpha            : h1 weight in improved_a_star.
        beta             : h2 weight in improved_a_star.
        epsilon          : roulette-wheel smoothing constant.

    Outputs: 
        best_individual  : list of K paths (None if task has no route).
        H_opt            : (N, N) connection matrix, passed as H_prev next snapshot.
        best_fitness     : fitness score of best individual found.
    """


    # evaluation function, we use this to evaluate the individual generated 
    def _evaluate(ind):
        return _evaluate_individual(
            ind, tasks, B_snapshot, D_cost, H_prev,
            graph_manager, weights, p_coefficients, M,
        )

    if positions_c is not None:
        diff = positions_c[:, np.newaxis, :] - positions_c[np.newaxis, :, :]
        heuristic_dist = np.linalg.norm(diff, axis=2)  
    else:
        heuristic_dist = D_cost  

    # line 1: initialize population
    population = []
    for _ in range(population_size):
        individual = [
            improved_a_star(
                int(t[0]), int(t[1]),
                neighbors, costs, heuristic_dist,
                alpha, beta, epsilon,
            )
            for t in tasks
        ]
        population.append(individual)

    # keep track of best individual and best_fitness 
    best_individual = None
    best_fitness = float("-inf")

    # line 2: loop for maximum number of evolutions 
    for _ in range(max_iterations):

        # lines 3-5: uses _evaluate function 
        fitnesses = []
        for ind in population:
            fitnesses.append(_evaluate(ind))
       
        # track best individuals for elitism. If this generations best individual beats the last generations individual, we can 
        # set it equal to the best indivdual with the best fitness
        gen_best = max(range(population_size), key=lambda i: fitnesses[i])
        if fitnesses[gen_best] > best_fitness:
            best_fitness = fitnesses[gen_best]

            # clone all paths to best_individual
            best_individual = [
                p[:] if p is not None else None for p in population[gen_best]
            ]

        # heredity (crossover and mutation: line 6)
        offspring = []
        while len(offspring) < population_size:
            p1 = _tournament_select(population, fitnesses)
            p2 = _tournament_select(population, fitnesses)
            c1, c2 = _crossover(p1, p2, eta_c)
            offspring.append(
                _mutate(c1, tasks, neighbors, costs, heuristic_dist, eta_m, alpha, beta, epsilon)
            )
            offspring.append(
                _mutate(c2, tasks, neighbors, costs, heuristic_dist, eta_m, alpha, beta, epsilon)
            )
        offspring = offspring[:population_size]

        # Line 7: recalculate fitness for offspring
        offspring_fitnesses = [_evaluate(ind) for ind in offspring]

        # Lines 8-9: combining child and parent, and cutting away worse half and keeping best half 
        combined = sorted(
            zip(population + offspring, fitnesses + offspring_fitnesses),
            key=lambda x: x[1],
            reverse=True,
        )
        population = [ind for ind, _ in combined[:population_size]]

    
    final_fitnesses = [_evaluate(ind) for ind in population]
    final_best_idx = max(range(population_size), key=lambda i: final_fitnesses[i])
    if final_fitnesses[final_best_idx] > best_fitness:
        best_fitness = final_fitnesses[final_best_idx]
        best_individual = population[final_best_idx]

    H_opt = _build_H(best_individual, graph_manager.num_nodes) # topology schedule

    return best_individual, H_opt, best_fitness

