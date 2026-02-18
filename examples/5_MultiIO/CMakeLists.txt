cmake_minimum_required(VERSION 3.18...3.31)
# policy CMP0076 - target_sources source files are relative to file where
# target_sources is run
cmake_policy(SET CMP0076 NEW)

set(PROJECT_NAME MultiIONetExample)

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
add_executable(multiionet_infer_fortran multiionet_infer_fortran.f90)
target_link_libraries(multiionet_infer_fortran PRIVATE FTorch::ftorch)

# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)

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
    set(MULTIIO_DEVICE_TYPE "cpu")
  else()
    string(TOLOWER "${GPU_DEVICE}" MULTIIO_DEVICE_TYPE)
  endif()

  # 1. Check the PyTorch model runs and its outputs meet expectations
  add_test(NAME example_multiio_multiionet
           COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/multiionet.py)

  # 2. Check the model is saved to file in the expected location with the
  #   pt2ts.py script
  add_test(
    NAME example_multiio_pt2ts
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/pt2ts.py --filepath
            ${PROJECT_BINARY_DIR} --device_type ${MULTIIO_DEVICE_TYPE}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

  # 3. Check the model can be loaded from file and run in Python and that its
  #   outputs meet expectations
  add_test(
    NAME example_multiio_multiionet_infer_python
    COMMAND
      ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/multiionet_infer_python.py
      --filepath ${PROJECT_BINARY_DIR} --device_type ${MULTIIO_DEVICE_TYPE}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

  # 4. Check the model can be loaded from file and run in Fortran and that its
  #   outputs meet expectations
  add_test(
    NAME example_multiio_multiionet_infer_fortran
    COMMAND
      multiionet_infer_fortran
      ${PROJECT_BINARY_DIR}/saved_multiio_model_${MULTIIO_DEVICE_TYPE}.pt
      ${GPU_DEVICE_CODE}
      # Command line arguments: model file, device type
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(
    example_multiio_multiionet_infer_fortran PROPERTIES PASS_REGULAR_EXPRESSION
    "MultiIO example ran successfully")
endif()
