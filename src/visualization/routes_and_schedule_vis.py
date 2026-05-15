import json
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

NUM_PLANES     = 7
SATS_PER_PLANE = 14

HDF5_PATH    = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "simulation_data.h5")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "results_1778805776.json")
IMAGES_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "images")


def _eci_to_latlon(pos):
    """ECI (x,y,z) km → (lat_deg, lon_deg) for cylindrical projection."""
    r   = np.linalg.norm(pos, axis=1, keepdims=False)
    lat = np.degrees(np.arcsin(np.clip(pos[:, 2] / r, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(pos[:, 1], pos[:, 0]))
    return lat, lon


def plot_routing_topology(H_opt, positions_t, tasks, snapshot_idx=0):
    lat, lon = _eci_to_latlon(positions_t)
    H        = np.array(H_opt)
    n_active = int(H.sum())

    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # ── background grid ──────────────────────────────────────────────────────
    ax.set_xlim(-185, 185)
    ax.set_ylim(-95, 95)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 15))
    ax.grid(True, color="#dddddd", linewidth=0.5, zorder=0)
    ax.axhline(0, color="#aaaaaa", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#aaaaaa", linewidth=0.7, zorder=0)

    # ── H_opt active routing links ───────────────────────────────────────────
    for i, j in np.argwhere(H):
        # skip antimeridian-crossing links to avoid ugly wrap-around lines
        if abs(lon[i] - lon[j]) > 180:
            continue
        ax.plot(
            [lon[i], lon[j]], [lat[i], lat[j]],
            color="black", linewidth=1.2, solid_capstyle="round", zorder=2,
        )

    # ── satellites ───────────────────────────────────────────────────────────
    src_nodes  = {int(t[0]) for t in tasks}
    dst_nodes  = {int(t[1]) for t in tasks}
    task_nodes = src_nodes | dst_nodes

    plane_cmap = plt.colormaps["tab10"]
    for sat_id in range(NUM_PLANES * SATS_PER_PLANE):
        plane      = sat_id // SATS_PER_PLANE
        edge_color = plane_cmap(plane / NUM_PLANES)
        face_color = "#FFE600" if sat_id in task_nodes else "white"
        ax.plot(
            lon[sat_id], lat[sat_id],
            "o",
            markersize=14,
            markerfacecolor=face_color,
            markeredgecolor=edge_color,
            markeredgewidth=1.8,
            zorder=3,
        )
        ax.text(
            lon[sat_id], lat[sat_id], str(sat_id),
            ha="center", va="center",
            fontsize=4.5, fontweight="bold", color="#111111",
            zorder=4,
        )

    ax.set_title(
        f"Walker-Delta Routing Topology — Snapshot {snapshot_idx}  "
        f"({n_active} active H_opt links)",
        fontsize=13, pad=10,
    )
    ax.set_xlabel("ECI Longitude (deg)", fontsize=10)
    ax.set_ylabel("Geocentric Latitude (deg)", fontsize=10)

    plt.tight_layout()
    os.makedirs(IMAGES_DIR, exist_ok=True)
    out_path = os.path.join(IMAGES_DIR, f"routing_topology_snapshot_{snapshot_idx:03d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.show()
    plt.close(fig)


def print_best_individual(best_individual, snapshot_idx=0):
    print(f"\n=== Best Individual — Snapshot {snapshot_idx} ===")
    routed = 0
    for k, path in enumerate(best_individual):
        if path is not None:
            nodes_str = " → ".join(str(n) for n in path)
            print(f"  Task {k:2d} ({len(path)-1} hops): {nodes_str}")
            routed += 1
        else:
            print(f"  Task {k:2d}: [unrouted]")
    print(f"  {routed}/{len(best_individual)} tasks routed")


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    with h5py.File(HDF5_PATH, "r") as hf:
        positions = hf["positions"][:]          # (T, N, 3)
        tasks     = hf["tasks"][:]              # (K, 3): src, dst, priority

    for snap in data["snapshots"]:
        idx             = snap["snapshot"]
        H_opt           = snap["H_opt"]
        best_individual = snap["best_individual"]
        positions_t     = positions[idx]        # (N, 3) for this snapshot

        print_best_individual(best_individual, idx)
        plot_routing_topology(H_opt, positions_t, tasks, idx)


if __name__ == "__main__":
    main()
