#include "constraints.hpp"
#include <algorithm>
#include <cmath>

bool constraints::is_link_blocked(const Eigen::Vector3d& sat_pos_1, const Eigen::Vector3d& sat_pos_2, double R_e, double atm_margin){
    
    Eigen::Vector3d d = sat_pos_2 - sat_pos_1;
    double t = -sat_pos_1.dot(d) / d.dot(d);
    t = std::clamp(t, 0.0, 1.0);
    Eigen::Vector3d closest = sat_pos_1 + t * d;

    double dist = closest.norm();

    if (dist < R_e + atm_margin){
        return true; // link is blocked, cannot send signal 
    } else {
        return false;
    }
}

double constraints::compute_B(double sigma_N2, double lambda, double distance_ij, double R, double P_T, double G_T, double G_R, double eta_T, double eta_R){
    // variable used to calculate the CDF of S (stochastic link failure probability)
    double B = 1.0/(2.0*sigma_N2)*std::pow((lambda/(4*M_PI*distance_ij)),4)*std::pow((R*P_T*G_T*G_R*eta_T*eta_R),2); 
    return B;
}

double constraints::compute_link_failure_probability(double a, double B, double G_T, double sigma_mod){
    // (stochastic link failure probability)
    double b_ij = a / B * std::exp(1.0/(4.0*G_T*std::pow(sigma_mod,2)));
    return b_ij;  
}