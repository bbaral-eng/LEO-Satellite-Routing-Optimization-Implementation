cmake_minimum_required(VERSION 3.12)
project(leo_routing_optimizer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(pybind11 REQUIRED)
pybind11_add_module(core_engine 
    src/core_engine/bindings.cpp
    src/core_engine/src/pathfinding.cpp
    src/core_engine/src/beckmann_math.cpp
    src/core_engine/src/constraints.cpp
)
target_include_directories(core_engine PRIVATE 
    src/core_engine/include
)