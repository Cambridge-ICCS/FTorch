cmake_minimum_required(VERSION 3.18...3.31)
# policy CMP0076 - target_sources source files are relative to file where
# target_sources is run
cmake_policy(SET CMP0076 NEW)

set(PROJECT_NAME MultiGPUExample)

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

# Map library build device option (GPU_DEVICE) to runtime strings used by
# Python/Fortran test drivers and to select the correct saved model filename.
#
# * Root CMake uses: NONE | CUDA | HIP | XPU | MPS
# * Python scripts expect lowercase: cpu | cuda | hip | xpu | mps
# * Fortran program expects corresponding enums, i.e., GPU_DEVICE_CODE
if("${GPU_DEVICE}" STREQUAL "")
  set(GPU_DEVICE "NONE")
  set(GPU_DEVICE_CODE "0")
endif()
if("${GPU_DEVICE}" STREQUAL "NONE")
  set(MULTIGPU_DEVICE_TYPE "cpu")
else()
  string(TOLOWER "${GPU_DEVICE}" MULTIGPU_DEVICE_TYPE)
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

if("${MULTI_GPU}" STREQUAL "ON")
  set(NUM_DEVICES 2)
else()
  set(NUM_DEVICES 1)
endif()

# Fortran example
add_executable(multigpu_infer_fortran multigpu_infer_fortran.f90)
target_link_libraries(multigpu_infer_fortran PRIVATE FTorch::ftorch)

# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)

  # 1. Check the PyTorch model runs on a CUDA device and its outputs meet
  # expectations
  add_test(NAME example_multigpu_simplenet
           COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/simplenet.py
                   --device_type ${MULTIGPU_DEVICE_TYPE})

  # 2 Check the model is saved to file in the expected location with the
  # pt2ts.py script
  add_test(
    NAME example_multigpu_pt2ts
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/pt2ts.py
            --filepath ${PROJECT_BINARY_DIR}
            --device_type ${MULTIGPU_DEVICE_TYPE}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

  # 3. Check the model can be loaded from file and run on two CUDA devices in
  # Python and that its outputs meet expectations
  add_test(
    NAME example_multigpu_infer_python
    COMMAND ${Python_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/multigpu_infer_python.py
            --filepath ${PROJECT_BINARY_DIR}
            --device_type ${MULTIGPU_DEVICE_TYPE}
            --num_devices ${NUM_DEVICES}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

  # 4. Check the model can be loaded from file and run on two CUDA devices in
  # Fortran and that its outputs meet expectations
  add_test(
    NAME example_multigpu_infer_fortran
    COMMAND multigpu_infer_fortran
    ${PROJECT_BINARY_DIR}/saved_multigpu_model_${MULTIGPU_DEVICE_TYPE}.pt
    ${GPU_DEVICE_CODE} ${NUM_DEVICES}
    # Command line arguments for model file, device type, num devices
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(
    example_multigpu_infer_fortran PROPERTIES PASS_REGULAR_EXPRESSION
    "MultiGPU example ran successfully")
endif()
