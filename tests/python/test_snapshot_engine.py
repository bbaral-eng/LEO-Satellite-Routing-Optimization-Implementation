import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import math
import h5py
import numpy as np
import core_engine

from src.routing.snapshot_engine import SnapshotEngine

CONFIG = "configs/constellation.yaml"
HDF5   = "data/processed/simulation_data.h5"

# Parameters from constellation.yaml 
G_T      = 10 ** (106.3 / 10)
G_R      = 10 ** (118.9 / 10)
P_T      = 10e-3
N_T, N_R = 0.5, 0.4
LAM      = 1064e-9
R_RESP   = 0.6003
SIGMA_N2 = 4.3e-12
A_THRESH = 1e-6

# C++ binding smoke tests

def test_beckmann_sigma_symmetric():
    """With mu=0 and equal sigmas, sigma_mod == sigma_x == 0.75 (Eq. 7)."""
    sigma_mod = core_engine.calculate_beckmann_sigma(0.0, 0.75, 0.0, 0.75)
    assert math.isclose(sigma_mod, 0.75, rel_tol=1e-9)


def test_link_not_blocked_same_side():
    sat1 = np.array([7371.0,  100.0, 0.0])
    sat2 = np.array([7371.0, 4000.0, 0.0])
    assert not core_engine.is_link_blocked(sat1, sat2, 6371.0, 100.0)


def test_link_blocked_opposite_poles():
    sat1 = np.array([0.0, 0.0,  7371.0])
    sat2 = np.array([0.0, 0.0, -7371.0])
    assert core_engine.is_link_blocked(sat1, sat2, 6371.0, 100.0)


def test_link_failure_probability_chain():
    """Full B -> b_ij chain at d_max=5000 km. Result must be a valid probability."""
    sigma_mod = core_engine.calculate_beckmann_sigma(0.0, 0.75, 0.0, 0.75)
    B = core_engine.compute_B(SIGMA_N2, LAM, 5000e3, R_RESP, P_T, G_T, G_R, N_T, N_R)
    assert B > 0.0
    b_ij = core_engine.compute_link_failure_probability(A_THRESH, B, G_T, sigma_mod)
    assert 0.0 < b_ij < 1.0

