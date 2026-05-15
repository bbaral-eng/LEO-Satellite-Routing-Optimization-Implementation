import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import time
import yaml
import numpy as np
from src.routing.graph_manager import GraphManager
from src.optimization.genetic_algo import GA_IAS, _build_H
from src.optimization.fitness_metrics import compute_psi, compute_f1, compute_f3, compute_objective

H5_PATH     = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "simulation_data.h5")
CONST_CFG   = os.path.join(os.path.dirname(__file__), "..", "configs", "constellation.yaml")
GA_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "ga_params.yaml")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_ga_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    p     = raw["GA-AS_Params"]
    astar = p["A_star_Parameters"]
    return {
        "population_size": int(p["NP"]),
        "max_iterations":  int(p["max_iterations"]),
        "eta_m":  float(p["eta_m"]),
        "eta_c":  float(p["eta_c"]),
        "eta_g":  float(p["eta_g"]),
        "M":      float(p["penalty_strategy"]["M"]),
        "weights": {
            "w1": float(p["multi_objective_weights"]["w1"]),
            "w2": float(p["multi_objective_weights"]["w2"]),
            "w3": float(p["multi_objective_weights"]["w3"]),
        },
        "alpha":   float(astar["heuristic_actual_cost_weight"]),
        "beta":    float(astar["heuristic_estimated_cost_weight"]),
        "epsilon": float(astar["roullete_smoothing_epsilon"]),
    }


def run_snapshot(c, gm, tasks, H_prev, cfg, p_coefficients):
    neighbors, costs = gm.get_snapshot_graph(c)
    D_cost     = gm.D_tensor[c]
    B_snapshot = gm.B_tensor[c]

    t0 = time.time()
    best_individual, H_opt, best_fitness = GA_IAS(
        tasks          = tasks,
        neighbors      = neighbors,
        costs          = costs,
        D_cost         = D_cost,
        B_snapshot     = B_snapshot,
        graph_manager  = gm,
        H_prev         = H_prev,
        weights        = cfg["weights"],
        p_coefficients = p_coefficients,
        population_size = cfg["population_size"],
        max_iterations  = cfg["max_iterations"],
        eta_m    = cfg["eta_m"],
        eta_c    = cfg["eta_c"],
        eta_g    = cfg["eta_g"],
        M        = cfg["M"],
        alpha    = cfg["alpha"],
        beta     = cfg["beta"],
        epsilon  = cfg["epsilon"],
    )
    elapsed = time.time() - t0

    # recompute f1/f2/f3 breakdown for the best individual
    priorities  = [float(t[2]) for t in tasks]
    valid_paths = [p for p in best_individual if p is not None]
    valid_prios = [priorities[k] for k, p in enumerate(best_individual) if p is not None]

    f1 = compute_f1(valid_paths, B_snapshot, valid_prios) if valid_paths else 0.0

    H_curr = _build_H(best_individual, gm.num_nodes)
    NS     = ((H_curr == 1) & (H_prev == 0)).astype(int)
    f2     = float(np.sum(NS))

    f3 = compute_f3(valid_paths, D_cost) if valid_paths else 0.0

    # per-task execution probability (psi_k)
    task_psis = []
    for path in best_individual:
        if path is not None and len(path) > 1:
            task_psis.append(round(compute_psi(path, B_snapshot), 6))
        else:
            task_psis.append(None)

    n_routed = sum(1 for p in best_individual if p is not None)

    print(f"\n--- Snapshot {c:3d}  ({elapsed:.1f}s) ---")
    print(f"  Best fitness F : {best_fitness:.6f}")
    print(f"  f1 (revenue)   : {f1:.6f}")
    print(f"  f2 (new links) : {f2:.0f}")
    print(f"  f3 (distance)  : {f3:.2f} km")
    print(f"  Tasks routed   : {n_routed}/{len(tasks)}")
    print(f"  Task psi       : {task_psis}")

    return {
        "snapshot":        c,
        "best_fitness":    best_fitness,
        "f1": f1, "f2": f2, "f3": f3,
        "n_routed":        n_routed,
        "n_tasks":         len(tasks),
        "task_psis":       task_psis,
        "elapsed_sec":     round(elapsed, 2),
        "best_individual": [p[:] if p is not None else None for p in best_individual],
        "H_opt":           H_opt.tolist(),
    }, H_opt


def main():
    parser = argparse.ArgumentParser(
        description="GA-AS routing optimizer for 98-node LEO constellation"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all 100 snapshots (default: snapshot 0 only)",
    )
    parser.add_argument(
        "--first", type=int, default=None, metavar="N",
        help="Run first N snapshots (0 through N-1)",
    )
    args = parser.parse_args()

    gm  = GraphManager(H5_PATH, CONST_CFG)
    cfg = load_ga_config(GA_CFG_PATH)

    tasks          = gm.tasks
    # p1 scales f1 (O~10) to ~500; p2 moderates f2 (O~50); p3 shrinks f3 (O~180,000) to ~180
    p_coefficients = {"p1": 1, "p2": 1, "p3": 4.5e-5}

    if args.all:
        snapshots   = range(gm.num_snapshots)
        snap_label  = "all"
    elif args.first is not None:
        snapshots   = range(min(args.first, gm.num_snapshots))
        snap_label  = f"first {args.first}"
    else:
        snapshots   = [0]
        snap_label  = "0 only"

    print(
        f"GA-AS | nodes={gm.num_nodes}  tasks={len(tasks)}  "
        f"pop={cfg['population_size']}  gens={cfg['max_iterations']}  "
        f"snapshots={snap_label}"
    )

    H_prev      = np.zeros((gm.num_nodes, gm.num_nodes), dtype=int)
    all_results = []

    for c in snapshots:
        result, H_prev = run_snapshot(c, gm, tasks, H_prev, cfg, p_coefficients)
        all_results.append(result)

    # summary statistics
    fitnesses = [r["best_fitness"] for r in all_results]
    f1s = [r["f1"] for r in all_results]
    f2s = [r["f2"] for r in all_results]
    f3s = [r["f3"] for r in all_results]

    print("\n_____________ SUMMARY _____________")
    print(f"  Snapshots run  : {len(all_results)}")
    print(f"  Avg fitness    : {np.mean(fitnesses):.6f}")
    print(f"  Max fitness    : {np.max(fitnesses):.6f}")
    print(f"  Min fitness    : {np.min(fitnesses):.6f}")
    print(f"  Avg f1         : {np.mean(f1s):.6f}")
    print(f"  Avg f2         : {np.mean(f2s):.2f}")
    print(f"  Avg f3         : {np.mean(f3s):.2f} km")
    print(f"  Total time     : {sum(r['elapsed_sec'] for r in all_results):.1f}s")

    # write results JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts       = int(time.time())
    out_path = os.path.join(RESULTS_DIR, f"results_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "config":  {**{k: v for k, v in cfg.items() if k != "weights"}, "p_coefficients": p_coefficients},
                "weights": cfg["weights"],
                "snapshots": all_results,
            },
            f,
            indent=2,
        )
    print(f"\n  Results written: {out_path}")


if __name__ == "__main__":
    main()
