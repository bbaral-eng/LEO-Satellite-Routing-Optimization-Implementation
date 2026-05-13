# necessary libraries 
import numpy as np 
import yaml 
import h5py
import os 


class SnapshotEngine:
    def __init__(self, config_path="configs/constellation.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        walker   = config["Network_Size"]["walker_blueprint"]
        self.isll     = config["ISLL_Model_Parameters"]
        self.pointing = config["Pointing_Error_Parameters"]
        self.ops      = config["Operational_Constraints"]

        self.NUM_PLANES     = walker["num_planes"]
        self.SATS_PER_PLANE = walker["sats_per_plane"]
        self.NUM_SATS       = self.NUM_PLANES * self.SATS_PER_PLANE
        self.ALT_KM         = walker["altitude_km"]
        self.INC_DEG        = walker["inclination_deg"]
        self.F_PHASE        = walker["f_inter_plane_phase"]
        self.TIME_STEPS     = config["Network_Size"]["Time_Steps"]
        self.DT_SEC         = config["Network_Size"]["t"]

        self.R_EARTH_KM = self.ops["earth_radius_km"]
        self.MU         = 3.986e14
        self.R_ORBIT_KM = self.R_EARTH_KM + self.ALT_KM
        self.R_ORBIT_M  = self.R_ORBIT_KM * 1000.0

        # linearized G_T and G_R
        self.G_T = 10 ** (self.isll["G_t"] / 10)
        self.G_R = 10 ** (self.isll["G_r"] / 10)
        self.ORBITAL_PERIOD_SEC = 2 * np.pi * np.sqrt(self.R_ORBIT_M**3 / self.MU)

    def compute_satellite_positions(self):
        """
        Compute ECI (x, y, z) positions in km for all satellites at every time step.
        Returns:
            positions: numpy array of shape (TIME_STEPS, NUM_SATS, 3)
        """
        positions = np.zeros((self.TIME_STEPS, self.NUM_SATS, 3))

        inc_rad = np.radians(self.INC_DEG)
        cos_inc = np.cos(inc_rad)
        sin_inc = np.sin(inc_rad)

        for p in range(self.NUM_PLANES):
            # RAAN: evenly space the orbital planes around the equator
            raan_rad = np.radians((360.0 / self.NUM_PLANES) * p)
            cos_raan = np.cos(raan_rad)
            sin_raan = np.sin(raan_rad)

            for s in range(self.SATS_PER_PLANE):
                sat_idx = p * self.SATS_PER_PLANE + s

                # Initial true anomaly: in-plane spacing + Walker inter-plane phasing
                nu0_rad = np.radians(
                    (360.0 / self.SATS_PER_PLANE) * s +
                    (360.0 / self.NUM_SATS) * self.F_PHASE * p
                )

                for c in range(self.TIME_STEPS):
                    # Advance along orbit with time
                    nu_rad = nu0_rad + (2 * np.pi / self.ORBITAL_PERIOD_SEC) * (c * self.DT_SEC)

                    cos_nu = np.cos(nu_rad)
                    sin_nu = np.sin(nu_rad)

                    # ECI rotation: orbital plane → inertial frame
                    x = self.R_ORBIT_KM * (cos_raan * cos_nu - sin_raan * sin_nu * cos_inc)
                    y = self.R_ORBIT_KM * (sin_raan * cos_nu + cos_raan * sin_nu * cos_inc)
                    z = self.R_ORBIT_KM * (sin_inc * sin_nu)

                    positions[c, sat_idx] = [x, y, z]

        return positions

    def compute_tensors(self, positions):
        """
        For each time step, compute the A, B, D tensors based on satellite positions and system parameters.

        returns: 
            A_tensor[c, i, j] = 1 if a physical link is possible between satellites i and j at time step c, else 0.
            B_tensor[c, i, j] = probability of link failure between satellites i and j at time step c, based on ISLL model and pointing errors.
            D_tensor[c, i, j] = physical distance in km between satellites 

        """
        A_tensor = np.zeros((self.TIME_STEPS, self.NUM_SATS, self.NUM_SATS)) # Adjacency tensor, indicates if physical link is possible  
        B_tensor = np.zeros((self.TIME_STEPS, self.NUM_SATS, self.NUM_SATS)) # Stochastic failure tensor, stores probability of failure 
        D_tensor = np.zeros((self.TIME_STEPS, self.NUM_SATS, self.NUM_SATS)) # Cost/Distance tensor, stores physical distance between i,j satellites 

        D_MAX_KM = self.ops["d_max"]
        R_E_KM   = self.ops["earth_radius_km"]
        ATM_KM   = self.ops["atmosphere_margin_km"]

        # sigma_mod = core_engine.calculate_beckmann_sigma(
        #     self.pointing["mu_x"], self.pointing["sigma_x"],
        #     self.pointing["mu_y"], self.pointing["sigma_y"]
        # )

        for c in range(self.TIME_STEPS):
            for i in range(self.NUM_SATS):
                for j in range(self.NUM_SATS):
                    if i == j:
                        continue # satellite does not need to communicate with itself 

                    pos_i   = positions[c, i]
                    pos_j   = positions[c, j]
                    dist_km = np.linalg.norm(pos_j - pos_i)

                    # max communication range 
                    if dist_km > D_MAX_KM:
                        continue

                    # earth occultation, checking if Earth blocks link 
                    # blocked = core_engine.is_link_blocked(pos_i, pos_j, R_E_KM, ATM_KM)
                    # if blocked:
                    #     continue

                    A_tensor[c, i, j] = 1.0
                    D_tensor[c, i, j] = dist_km

                    dist_m = dist_km * 1000.0

                    # B = core_engine.compute_B(
                    #     self.isll["noise_variance"], self.isll["lambda"], dist_m,
                    #     self.isll["R"], self.isll["P_t"], self.G_T, self.G_R,
                    #     self.isll["N_t"], self.isll["N_r"]
                    # )
                    # B_tensor[c, i, j] = core_engine.compute_link_failure_probability(
                    #     self.isll["a"], B, self.G_T, sigma_mod
                    # )

        return A_tensor, B_tensor, D_tensor

    def generate_routing_tasks(self, num_tasks=20):
        """
        Generate a list of routing tasks for the satellite network.

        returns:
            tasks: list of tuples (src, dst, priority) 
            src = source satellite index (0 to NUM_SATS-1)
            dst = destination satellite index (0 to NUM_SATS-1)
            priority = random float between 0.1 and 1.0 indicating task priority (higher is more urgent, duh)
        """
        np.random.seed(42)          # for reproducbility, remove later 
        tasks = []

        while len(tasks) < num_tasks:
            src = np.random.randint(0, self.NUM_SATS)
            dst = np.random.randint(0, self.NUM_SATS)

            if src == dst: 
                continue 

            priority = np.random.uniform(0.1, 1.0)
            priority = round(priority, 2)
            tasks.append((src, dst, priority))

        return tasks

    def save_snapshot(self, positions, A_tensor, B_tensor, D_tensor, tasks):
        """
        save computed snapshot to HDF5 file for later use. 
        """
        os.makedirs("data/processed", exist_ok=True)
        output_path = "data/processed/simulation_data.h5"

        with h5py.File(output_path, "w") as f:
            f.create_dataset("positions", data=positions)
            f.create_dataset("A_tensor", data=A_tensor)
            f.create_dataset("B_tensor", data=B_tensor)
            f.create_dataset("D_tensor", data=D_tensor)
            f.create_dataset("tasks", data=np.array(tasks, dtype=np.float64))

        print("data saved to:", output_path)

    def run(self):
        positions = self.compute_satellite_positions()
        A, B, D = self.compute_tensors(positions)
        tasks = self.generate_routing_tasks()
        self.save_snapshot(positions, A, B, D, tasks)


if __name__ == "__main__":
    engine = SnapshotEngine()
    engine.run()
