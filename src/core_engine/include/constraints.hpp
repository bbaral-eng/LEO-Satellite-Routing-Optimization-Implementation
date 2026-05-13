#pragma once
#include <Eigen/Dense>

namespace constraints {
    bool is_link_blocked(const Eigen::Vector3d& sat_pos_1, const Eigen::Vector3d& sat_pos_2, double R_e, double atm_margin);
    double compute_B(double sigma_N2, double lambda, double distance_ij, double R, double P_T, double G_T, double G_R, double eta_T, double eta_R);
    double compute_link_failure_probability(double a, double B, double G_T, double sigma_mod);
}

