import numpy as np
import h5py
import yaml

class GraphManager: 
    def __init__(self, h5_file, constellation):
        """
        Initializes the GraphManager by loading the constellation configuration and the graph data from the specified HDF5 file.
        """
        if isinstance(constellation, str):
            with open(constellation, "r") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = constellation

        with h5py.File(h5_file, "r") as f:
            self.positions = f["positions"][:]
            self.A_tensor  = f["A_tensor"][:]
            self.B_tensor  = f["B_tensor"][:]
            self.D_tensor  = f["D_tensor"][:]
            self.tasks     = f["tasks"][:]

        ops = cfg["Operational_Constraints"]
        self.o_i_max = int(ops["o_i_max"])
        self.l_i_max = int(ops["l_i_max"])

        self.num_snapshots = self.A_tensor.shape[0]
        self.num_nodes     = self.A_tensor.shape[1] 

    def get_snapshot_graph(self, c): 

        """
        Extracts adjacency tensor static graphs for snapshot index c (a point in time), returning a neighbors dict and a costs dict.
        - neighbors: dict mapping node index to list of neighboring node indices (i.e. outgoing edges)
        - costs: dict mapping (i, j) edge tuples to their corresponding cost from D_tensor
        """

        if not (0 <= c < self.num_snapshots):
            raise IndexError(f"Snapshot index {c} out of range [0, {self.num_snapshots})")

        edges = np.argwhere(self.A_tensor[c] == 1)
        neighbors = {i: [] for i in range(self.num_nodes)}
        costs = {}

        for i, j in edges:
            neighbors[i].append(int(j))
            costs[(int(i), int(j))] = float(self.D_tensor[c, i, j])

        return neighbors, costs

    def compute_switch_matrix(self, H_prev, H_curr):

        """
        This is just eq. 1 from the paper: penalizing new link switches by comparing the previous and current adjacency node state.
        """

        if H_prev.shape != H_curr.shape:
            raise ValueError(f"Shape mismatch: H_prev {H_prev.shape} vs H_curr {H_curr.shape}")
        return ((H_curr == 1) & (H_prev == 0)).astype(int)

    def check_degree_constraints(self, paths, n):

        """
        Counts how many per-node degree violations exist across all task path
        """

        out_degree = np.zeros(n, dtype=int)
        in_degree  = np.zeros(n, dtype=int)

        for path in paths:
            for hop in range(len(path) - 1):
                u = path[hop]           # current node, incoming edge
                v = path[hop + 1]       # next node, outgoing edge

                out_degree[u] += 1
                in_degree[v]  += 1

        violations = int(np.sum(out_degree > self.o_i_max) + np.sum(in_degree > self.l_i_max))
        return violations
