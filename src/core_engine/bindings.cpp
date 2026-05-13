#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "beckmann_math.hpp"
#include "constraints.hpp"
namespace py = pybind11;

PYBIND11_MODULE(core_engine, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("calculate_beckmann_sigma", &beckmann::calculate_beckmann_sigma, "A function that calculates ISLL vibrational error based on Beckmann distribution approx");
    m.def("is_link_blocked", &constraints::is_link_blocked, "A function that calculates whether a link between two satellites is blocked by Earth or not");
    m.def("compute_B", &constraints::compute_B, "A function that calculates B, which is the maximum possible received signal quality on this link ");
    m.def("compute_link_failure_probability", &constraints::compute_link_failure_probability, "A function that calculates ISLL link failure probability");
}