cmake_minimum_required(VERSION 3.15...3.31)
# policy CMP0076 - target_sources source files are relative to file where
# target_sources is run
cmake_policy(SET CMP0076 NEW)

set(PROJECT_NAME TrainingExample)

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
add_executable(training training.f90)
target_link_libraries(training PRIVATE FTorch::ftorch)

# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)

  # 1. Write the PyTorch model out and check it runs and its outputs meet
  #    expectations
  add_test(
    NAME example_training_simplenet
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/simplenet.py
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

  # 2. Run the pt2ts script to convert the model file to TorchScript format
  #    and run basic checks
  add_test(
    NAME example_training_pt2ts
    COMMAND pt2ts SimpleNet
                  --model_definition_file
                  ${PROJECT_SOURCE_DIR}/simplenet.py
                  --input_model_file
                  ${PROJECT_BINARY_DIR}/pytorch_simplenet_model_cpu.pt
                  --input_tensor_file
                  ${PROJECT_BINARY_DIR}/pytorch_simplenet_input_tensor_cpu.pt
                  --output_model_file
                  ${PROJECT_BINARY_DIR}/torchscript_simplenet_model_cpu.pt
                  --test)

  # 3. Check the Fortran training script runs successfully
  add_test(
    NAME example_training
    COMMAND training
      ${PROJECT_BINARY_DIR}/torchscript_simplenet_model_cpu.pt
      # Command line argument: model file
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(example_training
                       PROPERTIES PASS_REGULAR_EXPRESSION
                       "Fortran training example ran successfully")

  # 4. Plot results and check the plot file exists
  add_test(
    NAME example_training_plot
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/plot.py &&
            ${CMAKE_COMMAND} -E test_file ${PROJECT_BINARY_DIR}/losses.png
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
endif()
