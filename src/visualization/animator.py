# necessary libraries
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os

HDF5_PATH = "data/processed/simulation_data.h5"
GIF_PATH  = "assets/gifs/constellation.gif"
IMG_PATH  = "assets/images/adjacency_snapshots.png"

R_EARTH_KM    = 6371.0
NUM_PLANES    = 7
SATS_PER_PLANE = 14
NUM_SATS      = NUM_PLANES * SATS_PER_PLANE
AXIS_LIM      = 8500


# honest to god, this code is AI generated. I just cleaned it up and added some comments (mainly this one). 
# This code is so extremely basic and self-explanatory so feel free to read it top to bottom. 

def load_data():
    with h5py.File(HDF5_PATH, "r") as f:
        positions = f["positions"][:]    
        A         = f["A_tensor"][:]     
        B         = f["B_tensor"][:]     
    return positions, A, B


def _earth_surface():
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = R_EARTH_KM * np.outer(np.cos(u), np.sin(v))
    y = R_EARTH_KM * np.outer(np.sin(u), np.sin(v))
    z = R_EARTH_KM * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _plane_colors():
    cmap = plt.colormaps["tab10"]
    return [cmap(i) for i in range(NUM_PLANES)]


def _build_link_segments(pos_t, A_t, B_t):
    green, orange, red = [], [], []
    for i in range(NUM_SATS):
        for j in range(i + 1, NUM_SATS):
            if A_t[i, j] == 1.0:
                seg = [pos_t[i], pos_t[j]]
                b = B_t[i, j]
                if b < 0.01:
                    green.append(seg)
                elif b <= 0.1:
                    orange.append(seg)
                else:
                    red.append(seg)
    return green, orange, red


def _style_axes(ax, frame):
    ax.set_xlim(-AXIS_LIM, AXIS_LIM)
    ax.set_ylim(-AXIS_LIM, AXIS_LIM)
    ax.set_zlim(-AXIS_LIM, AXIS_LIM)
    ax.set_xlabel("x (km)", color="white", labelpad=6)
    ax.set_ylabel("y (km)", color="white", labelpad=6)
    ax.set_zlabel("z (km)", color="white", labelpad=6)
    ax.set_title(f"LEO Walker-Delta Constellation  |  t = {frame * 60} s", color="white", pad=10)
    ax.tick_params(colors="white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#1a1a3a")
    ax.yaxis.pane.set_edgecolor("#1a1a3a")
    ax.zaxis.pane.set_edgecolor("#1a1a3a")
    ax.grid(True, color="#222244", linewidth=0.4)


def animate(positions, A, B):
    ex, ey, ez   = _earth_surface()
    plane_colors = _plane_colors()

    fig = plt.figure(figsize=(9, 8), facecolor="#0a0a1a")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0a0a1a")

    def update(frame):
        ax.cla()
        ax.set_facecolor("#0a0a1a")

        # Earth
        ax.plot_surface(ex, ey, ez, color="steelblue", alpha=0.35,
                        linewidth=0, antialiased=True, zorder=0)

        # Satellites — one color per orbital plane
        pos_t = positions[frame]
        for p in range(NUM_PLANES):
            s = p * SATS_PER_PLANE
            e = s + SATS_PER_PLANE
            sat_xyz = pos_t[s:e]
            ax.scatter(sat_xyz[:, 0], sat_xyz[:, 1], sat_xyz[:, 2],
                       color=plane_colors[p], s=20, depthshade=False, zorder=5)

        # Inter-satellite links colored by failure probability
        green, orange, red = _build_link_segments(pos_t, A[frame], B[frame])
        for segs, col in [(green, "limegreen"), (orange, "orange"), (red, "red")]:
            if segs:
                lc = Line3DCollection(segs, colors=col, linewidths=0.5, alpha=0.5)
                ax.add_collection3d(lc)

        _style_axes(ax, frame)

    ani = animation.FuncAnimation(fig, update, frames=100, interval=100)
    os.makedirs(os.path.dirname(GIF_PATH), exist_ok=True)
    ani.save(GIF_PATH, writer=animation.PillowWriter(fps=10), dpi=110)
    plt.close(fig)
    print(f"GIF saved to {GIF_PATH}")


def plot_adjacency_snapshots(A):
    np.random.seed(7)
    timesteps = sorted(np.random.choice(100, 3, replace=False))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="white")
    fig.suptitle("Adjacency Matrix — 3 Random Snapshots", fontsize=13)

    for idx, (ax, t) in enumerate(zip(axes, timesteps)):
        im = ax.imshow(A[t], cmap="inferno", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"t = {t * 60} s", fontsize=10)
        ax.set_xlabel("Satellite j", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Satellite i", fontsize=9)
        else:
            ax.set_yticks([])

    fig.colorbar(im, ax=axes[-1], shrink=0.9, label="Link active")
    plt.tight_layout()
    os.makedirs(os.path.dirname(IMG_PATH), exist_ok=True)
    plt.savefig(IMG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved to {IMG_PATH}")


if __name__ == "__main__":
    positions, A, B = load_data()
    animate(positions, A, B)
    plot_adjacency_snapshots(A)
