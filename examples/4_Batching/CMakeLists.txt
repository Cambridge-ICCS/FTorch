cmake_minimum_required(VERSION 3.18...3.31)
# policy CMP0076 - target_sources source files are relative to file where
# target_sources is run
cmake_policy(SET CMP0076 NEW)

set(PROJECT_NAME BatchingNet)

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

# Install Python dependencies
find_package(Python COMPONENTS Interpreter REQUIRED)
if(NOT DEFINED ENV{VIRTUAL_ENV} AND NOT DEFINED ENV{CONDA_PREFIX})
  message(FATAL_ERROR "Please activate your virtualenv or conda environment")
endif()
execute_process(COMMAND ${Python_EXECUTABLE} -m pip install -r
                        ${PROJECT_SOURCE_DIR}/requirements.txt)

# Fortran example
add_executable(batchingnet_infer_fortran batchingnet_infer_fortran.f90)
target_link_libraries(batchingnet_infer_fortran PRIVATE FTorch::ftorch)

# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)

  # 1. Check the PyTorch model runs and its outputs meet expectations
  add_test(NAME example_batching_batching
           COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/batching_demo.py)

  # 2. Check the model is saved to file in the expected location with the
  #   pt2ts.py script
  add_test(
    NAME example_batching_pt2ts
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/pt2ts.py --filepath
            ${PROJECT_BINARY_DIR}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

  # 3. Load the model and run in Fortran checking that outputs meet expectations
  add_test(
    NAME example_batching_batching_infer_fortran
    COMMAND
      batchingnet_infer_fortran
      ${PROJECT_BINARY_DIR}/saved_batchingnet_model_cpu.pt
      # Command line argument: model file
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(
    example_batching_batching_infer_fortran PROPERTIES PASS_REGULAR_EXPRESSION
    "BatchingNet Fortran example ran successfully")
endif()

