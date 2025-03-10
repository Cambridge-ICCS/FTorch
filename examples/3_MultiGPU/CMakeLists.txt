cmake_minimum_required(VERSION 3.15...3.31)
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

find_package(FTorch)
message(STATUS "Building with Fortran PyTorch coupling")

include(CheckLanguage)
if("${GPU_DEVICE}" STREQUAL "CUDA")
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(ERROR "No CUDA support")
  endif()
endif()

# Fortran example
add_executable(multigpu_infer_fortran multigpu_infer_fortran.f90)
target_link_libraries(multigpu_infer_fortran PRIVATE FTorch::ftorch)

# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)

  if("${GPU_DEVICE}" STREQUAL "CUDA")
    # 1a. Check the PyTorch model runs on a CUDA device and its outputs meet
    # expectations
    add_test(NAME simplenet
             COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/simplenet.py
                     --device_type cuda)

    # 2a. Check the model is saved to file in the expected location with the
    # pt2ts.py script
    add_test(
      NAME pt2ts
      COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/pt2ts.py --device_type
              cuda --filepath ${PROJECT_BINARY_DIR}
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    # 3a. Check the model can be loaded from file and run on two CUDA devices in
    # Python and that its outputs meet expectations
    add_test(
      NAME multigpu_infer_python
      COMMAND ${Python_EXECUTABLE}
              ${PROJECT_SOURCE_DIR}/multigpu_infer_python.py --device_type cuda
              --filepath ${PROJECT_BINARY_DIR}
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    # 4a. Check the model can be loaded from file and run on two CUDA devices in
    # Fortran and that its outputs meet expectations
    add_test(
      NAME multigpu_infer_fortran
      COMMAND multigpu_infer_fortran cuda
              ${PROJECT_BINARY_DIR}/saved_multigpu_model_cuda.pt
      # Command line arguments for device type and model file
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    set_tests_properties(
      multigpu_infer_fortran PROPERTIES PASS_REGULAR_EXPRESSION
                                        "MultiGPU example ran successfully")
  endif()

  if("${GPU_DEVICE}" STREQUAL "XPU")
    # 1b. Check the PyTorch model runs on an XPU device and its outputs meet
    # expectations
    add_test(NAME simplenet
             COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/simplenet.py
                     --device_type xpu)

    # 2b. Check the model is saved to file in the expected location with the
    # pt2ts.py script
    add_test(
      NAME pt2ts
      COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/pt2ts.py --device_type
              xpu --filepath ${PROJECT_BINARY_DIR}
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    # 3b. Check the model can be loaded from file and run on two XPU devices in
    # Python and that its outputs meet expectations
    add_test(
      NAME multigpu_infer_python
      COMMAND ${Python_EXECUTABLE}
              ${PROJECT_SOURCE_DIR}/multigpu_infer_python.py --device_type xpu
              --filepath ${PROJECT_BINARY_DIR}
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    # 4b. Check the model can be loaded from file and run on two XPU devices in
    # Fortran and that its outputs meet expectations
    add_test(
      NAME multigpu_infer_fortran
      COMMAND multigpu_infer_fortran xpu
              ${PROJECT_BINARY_DIR}/saved_multigpu_model_xpu.pt
      # Command line arguments for device type and model file
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    set_tests_properties(
      multigpu_infer_fortran PROPERTIES PASS_REGULAR_EXPRESSION
                                        "MultiGPU example ran successfully")
  endif()

  if("${GPU_DEVICE}" STREQUAL "MPS")
    # 1c. Check the PyTorch model runs on an MPS device and its outputs meet
    # expectations
    add_test(NAME simplenet
             COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/simplenet.py
                     --device_type mps)
    # 2c. Check the model is saved to file in the expected location with the
    # pt2ts.py script
    add_test(
      NAME pt2ts
      COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/pt2ts.py --device_type
              mps --filepath ${PROJECT_BINARY_DIR}
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    # 3c. Check the model can be loaded from file and run on one MPS device in
    # Python and that its outputs meet expectations
    add_test(
      NAME multigpu_infer_python
      COMMAND ${Python_EXECUTABLE}
              ${PROJECT_SOURCE_DIR}/multigpu_infer_python.py --device_type mps
              --filepath ${PROJECT_BINARY_DIR}
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    # 4c. Check the model can be loaded from file and run on one MPS device in
    # Fortran and that its outputs meet expectations
    add_test(
      NAME multigpu_infer_fortran
      COMMAND multigpu_infer_fortran mps
              ${PROJECT_BINARY_DIR}/saved_multigpu_model_mps.pt
      # Command line arguments for device type and model file
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    set_tests_properties(
      multigpu_infer_fortran PROPERTIES PASS_REGULAR_EXPRESSION
                                        "MultiGPU example ran successfully")
  endif()
endif()
