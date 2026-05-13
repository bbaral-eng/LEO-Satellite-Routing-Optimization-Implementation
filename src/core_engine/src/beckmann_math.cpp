#include "beckmann_math.hpp"
#include <cmath> 

double beckmann::calculate_beckmann_sigma(double mu_x, double sigma_x, double mu_y, double sigma_y){

    // eq. 7 from the paper
    double numerator = (3*std::pow(mu_x, 2)*std::pow(sigma_x, 4))+(3*std::pow(mu_y, 2)*std::pow(sigma_y, 4))+std::pow(sigma_x,6)+std::pow(sigma_y,6);
    double sigma_mod = std::pow(numerator / 2.0, 1.0/6.0);

    return sigma_mod; 
} 