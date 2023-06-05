cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
#policy CMP0076 - target_sources source files are relative to file where target_sources is run
cmake_policy (SET CMP0076 NEW)

set(PROJECT_NAME ResNetExample)

project(${PROJECT_NAME} LANGUAGES Fortran)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

find_package(FTorch)
message(STATUS "Building with Fortran PyTorch coupling")

## C++ demo
#add_executable(ts_infer_cpp ts_inference.cpp)
#target_link_libraries(ts_infer_cpp ${TORCH_LIBRARIES})
#
## C demo
#add_executable(ts_infer_c ts_inference.c)
#target_link_libraries(ts_infer_c ${PROJECT_NAME}::${LIB_NAME})

# Fortran demo
add_executable(resnet_infer_fortran resnet_infer_fortran.f90)
target_link_libraries(resnet_infer_fortran PRIVATE FTorch::ftorch)
