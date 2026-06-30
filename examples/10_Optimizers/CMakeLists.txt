cmake_minimum_required(VERSION 3.15...3.31)
# policy CMP0076 - target_sources source files are relative to file where
# target_sources is run
cmake_policy(SET CMP0076 NEW)

set(PROJECT_NAME OptimizersExample)

project(${PROJECT_NAME} LANGUAGES Fortran)

# Build in Debug mode if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "" FORCE)
endif()

if(CMAKE_BUILD_TESTS)
  find_package(FTorch QUIET)
else()
  find_package(FTorch REQUIRED)
endif()
message(STATUS "Building with Fortran PyTorch coupling")

# Check Python virtual environment is active
find_package(Python COMPONENTS Interpreter REQUIRED)
if(NOT DEFINED ENV{VIRTUAL_ENV} AND NOT DEFINED ENV{CONDA_PREFIX})
  message(FATAL_ERROR "Please activate your virtualenv or conda environment")
endif()

# Fortran example
add_executable(optimizers optimizers.f90)
target_link_libraries(optimizers PRIVATE FTorch::ftorch)

# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)

  # 1. Check the Python Optimizers script runs successfully
  add_test(
    NAME example_optim_pyoptim
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/optimizers.py
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(example_optim_pyoptim PROPERTIES PASS_REGULAR_EXPRESSION
                       "Python optimizers example ran successfully")

  # 2. Check the Fortran Optimizers script runs successfully
  add_test(
    NAME example_optim_foptim
    COMMAND optimizers
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(example_optim_foptim PROPERTIES PASS_REGULAR_EXPRESSION
                       "Fortran optimizers example ran successfully")

  # 3. Plot results and verify Python and Fortran saved losses match
  add_test(
    NAME example_optim_plot_and_compare
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/plot_and_compare.py &&
            ${CMAKE_COMMAND} -E test_file ${PROJECT_BINARY_DIR}/losses.png
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(example_optim_plot_and_compare PROPERTIES
                       PASS_REGULAR_EXPRESSION
                       "SUCCESS: Python and Fortran optimizer results match")
endif()
