cmake_minimum_required(VERSION 3.18...3.31)
# policy CMP0076 - target_sources source files are relative to file where
# target_sources is run
cmake_policy(SET CMP0076 NEW)

set(PROJECT_NAME MultiGPUExample)

project(${PROJECT_NAME} LANGUAGES C CXX Fortran)

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

include(CheckLanguage)
if("${GPU_DEVICE}" STREQUAL "CUDA")
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(ERROR "No CUDA support")
  endif()
endif()
if("${GPU_DEVICE}" STREQUAL "HIP")
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
  else()
    message(WARNING "No HIP support")
  endif()
endif()

# Fortran example
add_executable(multigpu_infer_fortran multigpu_infer_fortran.f90)
target_link_libraries(multigpu_infer_fortran PRIVATE FTorch::ftorch)

# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)

  # Map library build device option (GPU_DEVICE) to runtime strings used by
  # Python/Fortran test drivers and to select the correct saved model filename.
  #
  # * Root CMake uses the following strings:
  #   NONE | CUDA | HIP | XPU | MPS
  # * The Python scripts and Fortran program expect the lowercase counterparts:
  #   cpu | cuda | hip | xpu | mps
  if("${GPU_DEVICE}" STREQUAL "")
    set(GPU_DEVICE "NONE")
  endif()
  if("${GPU_DEVICE}" STREQUAL "NONE")
    set(DEVICE_LOW "cpu")
  else()
    string(TOLOWER "${GPU_DEVICE}" DEVICE_LOW)
  endif()

  # Only use multiple GPUs if the MULTI_GPU option is enabled
  if("${MULTI_GPU}" STREQUAL "ON")
    set(NUM_DEVICES 2)
  else()
    set(NUM_DEVICES 1)
  endif()

  # 1. Write the PyTorch model out with the specified device type and check it
  #    runs on that device type and its outputs meet expectations
  add_test(NAME example_multigpu_simplenet_${DEVICE_LOW}
           COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/simplenet.py
                   --device_type ${DEVICE_LOW}
           WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

  # 2. Run the pt2ts script to convert the model file to TorchScript format
  #    and run basic checks
  add_test(
    NAME example_multigpu_pt2ts_${DEVICE_LOW}
    COMMAND
    pt2ts SimpleNet
          --model_definition_file
          ${PROJECT_SOURCE_DIR}/simplenet.py
          --input_model_file
          ${PROJECT_BINARY_DIR}/pytorch_simplenet_model_${DEVICE_LOW}.pt
          --input_tensor_file
          ${PROJECT_BINARY_DIR}/pytorch_simplenet_input_tensor_${DEVICE_LOW}.pt
          --output_model_file
          ${PROJECT_BINARY_DIR}/torchscript_simplenet_model_${DEVICE_LOW}.pt
          --test)

  # 3. Check the model can be loaded from file and run on the specified devices
  #    in Python and that its outputs meet expectations
  add_test(
    NAME example_multigpu_infer_python_${DEVICE_LOW}
    COMMAND ${Python_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/multigpu_infer_python.py
            --device_type ${DEVICE_LOW}
            --filepath ${PROJECT_BINARY_DIR}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

  # 4. Check the model can be loaded from file and run on the specified devices
  #    in Fortran and that its outputs meet expectations
  add_test(
    NAME example_multigpu_infer_fortran_${DEVICE_LOW}
    COMMAND multigpu_infer_fortran ${DEVICE_LOW}
            ${PROJECT_BINARY_DIR}/torchscript_multigpu_model_${DEVICE_LOW}.pt
    # Command line arguments for device type and model file
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(
    example_multigpu_infer_fortran_${DEVICE_LOW}
    PROPERTIES PASS_REGULAR_EXPRESSION "MultiGPU example ran successfully")
endif()
