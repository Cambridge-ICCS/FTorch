cmake_minimum_required(VERSION 3.18...3.31)
# policy CMP0076 - target_sources source files are relative to file where
# target_sources is run
cmake_policy(SET CMP0076 NEW)

set(PROJECT_NAME ResNetExample)

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
add_executable(resnet_infer_fortran resnet_infer_fortran.f90)
target_link_libraries(resnet_infer_fortran PRIVATE FTorch::ftorch)

# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)

  # 1. Check the PyTorch model runs and its outputs meet expectations
  add_test(
    NAME example_resnet_resnet18
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/resnet18.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

  # 2. Check the model is saved to file in the expected location with the
  #   pt2ts.py script
  add_test(
    NAME example_resnet_pt2ts
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/pt2ts.py --filepath
            ${PROJECT_BINARY_DIR}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

  # 3. Check the model can be loaded from file and run in Fortran and that its
  #   outputs meet expectations
  add_test(
    NAME example_resnet_resnet_infer_fortran
    COMMAND
      resnet_infer_fortran ${PROJECT_BINARY_DIR}/saved_resnet18_model_cpu.pt
      ${PROJECT_SOURCE_DIR}/data
      # Command line arguments: model file and data directory filepath
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(
    example_resnet_resnet_infer_fortran PROPERTIES PASS_REGULAR_EXPRESSION
    "ResNet18 example ran successfully")
endif()
