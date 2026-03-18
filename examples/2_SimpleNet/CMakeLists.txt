cmake_minimum_required(VERSION 3.18...3.31)
# policy CMP0076 - target_sources source files are relative to file where
# target_sources is run
cmake_policy(SET CMP0076 NEW)

set(PROJECT_NAME SimpleNetExample)

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
add_executable(simplenet_infer_fortran simplenet_infer_fortran.f90)
target_link_libraries(simplenet_infer_fortran PRIVATE FTorch::ftorch)

# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)

  # 1. Write the PyTorch model out and check it runs and its outputs meet
  #    expectations
  add_test(
    NAME example_simplenet_simplenet
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/simplenet.py
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

  # 2. Run the pt2ts script to convert the model file to TorchScript format
  #    and run basic checks
  add_test(
    NAME example_simplenet_pt2ts
    COMMAND pt2ts SimpleNet
                  --model_definition_file ${PROJECT_SOURCE_DIR}/simplenet.py
                  ${PROJECT_SOURCE_DIR}/saved_simplenet_model_cpu.pt
                  --input_tensor_file
                  ${PROJECT_SOURCE_DIR}/saved_simplenet_input_tensor_cpu.pt
                  --output_model_file
                  ${PROJECT_BINARY_DIR}/saved_simplenet_model_cpu.pt
                  --test)

  # 3. Check the model can be loaded from file and run in Python and that its
  #   outputs meet expectations
  add_test(
    NAME example_simplenet_simplenet_infer_python
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/simplenet_infer_python.py
            --filepath ${PROJECT_BINARY_DIR}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

  # 4. Check the model can be loaded from file and run in Fortran and that its
  #   outputs meet expectations
  add_test(
    NAME example_simplenet_simplenet_infer_fortran
    COMMAND
      simplenet_infer_fortran ${PROJECT_BINARY_DIR}/saved_simplenet_model_cpu.pt
      # Command line argument: model file
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(
    example_simplenet_simplenet_infer_fortran PROPERTIES PASS_REGULAR_EXPRESSION
    "SimpleNet example ran successfully")
endif()
