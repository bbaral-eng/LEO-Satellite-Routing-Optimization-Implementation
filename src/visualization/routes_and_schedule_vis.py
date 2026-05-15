import io
import json
import os
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

_HERE        = os.path.dirname(__file__)
HDF5_PATH    = os.path.join(_HERE, "..", "..", "data", "processed", "simulation_data.h5")
RESULTS_PATH = os.path.join(_HERE, "..", "..", "data", "processed", "results_1778821463.json")
IMAGES_DIR   = os.path.join(_HERE, "..", "..", "assets", "images")
GIFS_DIR     = os.path.join(_HERE, "..", "..", "assets", "gifs")

NUM_PLANES     = 7
SATS_PER_PLANE = 14

def _eci_to_latlon(positions_t):
    """ECI to lat long"""
    r   = np.linalg.norm(positions_t, axis=1)
    lat = np.degrees(np.arcsin(np.clip(positions_t[:, 2] / r, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(positions_t[:, 1], positions_t[:, 0]))
    return lat, lon


def _draw_link_wrapped(ax, lon_i, lat_i, lon_j, lat_j, **kwargs):
    """
    Draw a link on a cylindrical map, wrapping at the antimeridian
    """
    dlon = lon_j - lon_i
    if dlon > 180:
        dlon -= 360
    elif dlon < -180:
        dlon += 360

    lon_j_adj = lon_i + dlon
    artists   = []

    if lon_j_adj > 180:
        t     = (180.0 - lon_i) / dlon
        lat_x = lat_i + t * (lat_j - lat_i)
        artists.extend(ax.plot([lon_i,  180], [lat_i, lat_x], **kwargs))
        artists.extend(ax.plot([-180, lon_j], [lat_x, lat_j], **kwargs))
    elif lon_j_adj < -180:
        t     = (-180.0 - lon_i) / dlon
        lat_x = lat_i + t * (lat_j - lat_i)
        artists.extend(ax.plot([lon_i, -180], [lat_i, lat_x], **kwargs))
        artists.extend(ax.plot([180,  lon_j], [lat_x, lat_j], **kwargs))
    else:
        artists.extend(ax.plot([lon_i, lon_j_adj], [lat_i, lat_j], **kwargs))

    return artists



def _make_base_figure(positions_t, H_opt, tasks, best_individual):
    """
    base figure background 
    """
    lat, lon = _eci_to_latlon(positions_t)
    H        = np.array(H_opt)

    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.set_xlim(-185, 185)
    ax.set_ylim(-95, 95)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 15))
    ax.grid(True, color="#dddddd", linewidth=0.5, zorder=0)
    ax.axhline(0, color="#aaaaaa", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#aaaaaa", linewidth=0.7, zorder=0)

    for i, j in np.argwhere(H):
        _draw_link_wrapped(ax, lon[i], lat[i], lon[j], lat[j],
                           color="black", linewidth=1.2,
                           solid_capstyle="round", zorder=2)

    task_nodes = {int(tasks[k][col])
                  for k in range(len(tasks)) for col in (0, 1)}

    plane_cmap = plt.colormaps["tab10"]
    for sat_id in range(NUM_PLANES * SATS_PER_PLANE):
        plane = sat_id // SATS_PER_PLANE
        ax.plot(lon[sat_id], lat[sat_id], "o",
                markersize=14,
                markerfacecolor="#FFE600" if sat_id in task_nodes else "white",
                markeredgecolor=plane_cmap(plane / NUM_PLANES),
                markeredgewidth=1.8, zorder=3)
        ax.text(lon[sat_id], lat[sat_id], str(sat_id),
                ha="center", va="center",
                fontsize=4.5, fontweight="bold", color="#111111", zorder=4)

    ax.set_xlabel("ECI Longitude (deg)", fontsize=10)
    ax.set_ylabel("Geocentric Latitude (deg)", fontsize=10)

    return fig, ax, lat, lon


def plot_routing_topology(H_opt, positions_t, tasks, best_individual, snapshot_idx):
    """Save a static routing-topology PNG for one snapshot."""
    fig, ax, _, _ = _make_base_figure(positions_t, H_opt, tasks, best_individual)
    n_active = int(np.array(H_opt).sum())
    ax.set_title(
        f"Walker-Delta Routing Topology — Snapshot {snapshot_idx}  "
        f"({n_active} active H_opt links)",
        fontsize=13, pad=10,
    )
    os.makedirs(IMAGES_DIR, exist_ok=True)
    out = os.path.join(IMAGES_DIR, f"routing_topology_snapshot_{snapshot_idx:03d}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [static] Saved → {out}")



def generate_path_gif(H_opt, positions_t, tasks, best_individual, snapshot_idx):
    """
    Animate every routed task path edge-by-edge in thick cyan on top of the
    static base view.  Edges are drawn cumulatively within each task; the cyan
    overlay is cleared before the next task begins.
    """
    fig, ax, lat, lon = _make_base_figure(positions_t, H_opt, tasks, best_individual)
    title = ax.set_title(
        f"Snapshot {snapshot_idx} — Routing Animation", fontsize=13, pad=10
    )

    pil_frames: list[Image.Image] = []
    durations:  list[int]         = []

    def _capture(ms: int):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        buf.seek(0)
        pil_frames.append(Image.open(buf).copy())
        durations.append(ms)

    _capture(800)   # opening frame

    routed   = [(k, p) for k, p in enumerate(best_individual)
                if p is not None and len(p) >= 2]
    n_routed = len(routed)

    for task_num, (k, path) in enumerate(routed, start=1):
        edges        = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        path_artists = []

        for step, (a, b) in enumerate(edges, start=1):
            new = _draw_link_wrapped(ax, lon[a], lat[a], lon[b], lat[b],
                                     color="cyan", linewidth=5.0,
                                     solid_capstyle="round", zorder=5)
            path_artists.extend(new)
            title.set_text(
                f"Snapshot {snapshot_idx} — Task {task_num}/{n_routed}  "
                f"hop {step}/{len(edges)}  ({a} → {b})"
            )
            _capture(450)

        title.set_text(
            f"Snapshot {snapshot_idx} — Task {task_num}/{n_routed}  [complete]"
        )
        _capture(900)

        for artist in path_artists:
            artist.remove()

        if task_num < n_routed:
            title.set_text(f"Snapshot {snapshot_idx} — Routing Animation")
            _capture(300)

    title.set_text(f"Snapshot {snapshot_idx} — All {n_routed} tasks animated")
    _capture(1500)  # closing frame

    plt.close(fig)

    os.makedirs(GIFS_DIR, exist_ok=True)
    out = os.path.join(GIFS_DIR, f"routing_animation_snapshot_{snapshot_idx:03d}.gif")
    pil_frames[0].save(
        out,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=0,
    )
    print(f"  [gif]    Saved → {out}  ({len(pil_frames)} frames)")



def main():
    random.seed(7)   
    with open(RESULTS_PATH) as f:
        all_snaps = json.load(f)["snapshots"]

    with h5py.File(HDF5_PATH, "r") as hf:
        positions = hf["positions"][:]   # (T, N, 3)
        tasks_all = hf["tasks"][:]       # (T, K, 3) per-snapshot task sets

    chosen   = random.sample(all_snaps, k=2)
    gif_snap = random.choice(chosen)

    print(f"Chosen snapshots : {[s['snapshot'] for s in chosen]}")
    print(f"GIF snapshot     : {gif_snap['snapshot']}\n")

    for snap in chosen:
        idx             = snap["snapshot"]
        positions_t     = positions[idx]
        tasks           = tasks_all[idx]
        H_opt           = snap["H_opt"]
        best_individual = snap["best_individual"]
        plot_routing_topology(H_opt, positions_t, tasks, best_individual, idx)

    idx             = gif_snap["snapshot"]
    positions_t     = positions[idx]
    tasks           = tasks_all[idx]
    H_opt           = gif_snap["H_opt"]
    best_individual = gif_snap["best_individual"]

    print(f"\nGenerating GIF for snapshot {idx} …")
    generate_path_gif(H_opt, positions_t, tasks, best_individual, idx)

    print("\nDone.")


if __name__ == "__main__":
    main()
