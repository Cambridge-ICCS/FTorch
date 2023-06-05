cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
#policy CMP0076 - target_sources source files are relative to file where target_sources is run
cmake_policy (SET CMP0076 NEW)

set(PROJECT_NAME ResNetExample)

project(${PROJECT_NAME} LANGUAGES Fortran C)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

find_package(FTorch)
message(STATUS "Building with Fortran PyTorch coupling")

## C++ demo
#add_executable(resnet_infer_cpp resnet_infer_cpp.cpp)
#target_link_libraries(resnet_infer_cpp ${TORCH_LIBRARIES})
#
## C demo
#add_executable(resnet_infer_c resnet_infer_c.c)
#target_link_libraries(resnet_infer_c FTorch::ftorch)

# Fortran demo
add_executable(resnet_infer_fortran resnet_infer_fortran.f90)
target_link_libraries(resnet_infer_fortran PRIVATE FTorch::ftorch)
