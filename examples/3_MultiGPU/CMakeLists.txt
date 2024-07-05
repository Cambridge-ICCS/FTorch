cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
#policy CMP0076 - target_sources source files are relative to file where target_sources is run
cmake_policy (SET CMP0076 NEW)

set(PROJECT_NAME MultiGPUExample)

project(${PROJECT_NAME} LANGUAGES Fortran)

# Build in Debug mode if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

find_package(FTorch)
find_package(MPI REQUIRED)
message(STATUS "Building with Fortran PyTorch coupling")

# Fortran example
add_executable(simplenet_infer_fortran_gpu simplenet_infer_fortran_gpu.f90)
target_link_libraries(simplenet_infer_fortran_gpu PRIVATE FTorch::ftorch)
target_link_libraries(simplenet_infer_fortran_gpu PRIVATE MPI::MPI_Fortran)
