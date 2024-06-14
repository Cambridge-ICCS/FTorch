cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
#policy CMP0076 - target_sources source files are relative to file where target_sources is run
cmake_policy (SET CMP0076 NEW)

set(PROJECT_NAME SimpleNetExample)

project(${PROJECT_NAME} LANGUAGES Fortran)

# Build in Debug mode if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

find_package(FTorch)
message(STATUS "Building with Fortran PyTorch coupling")

# # Fortran example # TODO
# add_executable(simplenet_infer_fortran simplenet_infer_fortran.f90)
# target_link_libraries(simplenet_infer_fortran PRIVATE FTorch::ftorch)

# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)

  # 1. Check the PyTorch model runs and its outputs meet expectations
  add_test(NAME simplenet
    COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/simplenet.py)
  set_tests_properties(simplenet PROPERTIES PASS_REGULAR_EXPRESSION
    "0., 2., 4., 6.")
  set_tests_properties(simplenet PROPERTIES PASS_REGULAR_EXPRESSION
    "0., -3., -6., -9.")
endif()
