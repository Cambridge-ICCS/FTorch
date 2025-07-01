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

find_package(FTorch)
message(STATUS "Building with Fortran PyTorch coupling")

# Fortran example
add_executable(optimizers optimizers.f90)
target_link_libraries(optimizers PRIVATE FTorch::ftorch)

# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)

  # 1. Check the Python Optimizers script runs successfully
  add_test(
    NAME pyoptim
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/optimizers.py
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(pyoptim PROPERTIES PASS_REGULAR_EXPRESSION
                       "Python optimizers example ran successfully")

  # 2. Check the Fortran Optimizers script runs successfully
  add_test(
    NAME foptim
    COMMAND optimizers
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(foptim PROPERTIES PASS_REGULAR_EXPRESSION
                       "Fortran optimizers example ran successfully")

  # 3. Check the plotting script runs successfully
  add_test(
    NAME plot
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/plot_losses.py && ${CMAKE_COMMAND} -E test_file ${PROJECT_BINARY_DIR}/losses.png
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
endif()
