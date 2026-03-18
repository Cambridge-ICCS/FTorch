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

  if("${GPU_DEVICE}" STREQUAL "CUDA")
    # 1a. Write the PyTorch model out with the CUDA device type and check it
    #     runs on a CUDA device and its outputs meet expectations
    add_test(NAME example_multigpu_simplenet
             COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/simplenet.py
                     --device_type cuda)

    # 2a. Run the pt2ts script to convert the model file to TorchScript format
    #     and run basic checks
    add_test(
      NAME example_multigpu_pt2ts
      COMMAND pt2ts SimpleNet
                    --model_definition_file ${PROJECT_SOURCE_DIR}/simplenet.py
                    ${PROJECT_SOURCE_DIR}/saved_simplenet_model_cuda.pt
                    --input_tensor_file
                    ${PROJECT_SOURCE_DIR}/saved_simplenet_input_tensor_cuda.pt
                    --output_model_file
                    ${PROJECT_BINARY_DIR}/saved_simplenet_model_cuda.pt
                    --test)

    # 3a. Check the model can be loaded from file and run on two CUDA devices in
    # Python and that its outputs meet expectations
    add_test(
      NAME example_multigpu_multigpu_infer_python
      COMMAND ${Python_EXECUTABLE}
              ${PROJECT_SOURCE_DIR}/multigpu_infer_python.py --device_type cuda
              --filepath ${PROJECT_BINARY_DIR}
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    # 4a. Check the model can be loaded from file and run on two CUDA devices in
    # Fortran and that its outputs meet expectations
    add_test(
      NAME example_multigpu_infer_fortran
      COMMAND multigpu_infer_fortran cuda
              ${PROJECT_BINARY_DIR}/saved_multigpu_model_cuda.pt
      # Command line arguments for device type and model file
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    set_tests_properties(
      example_multigpu_infer_fortran PROPERTIES PASS_REGULAR_EXPRESSION
      "MultiGPU example ran successfully")
  endif()

  if("${GPU_DEVICE}" STREQUAL "HIP")
    # 1b. Write the PyTorch model out with the HIP device type and check it
    #     runs on a HIP device and its outputs meet expectations
    add_test(NAME simplenet
             COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/simplenet.py
                     --device_type hip)

    # 2b. Run the pt2ts script to convert the model file to TorchScript format
    #     and run basic checks
    add_test(
      NAME example_multigpu_pt2ts
      COMMAND pt2ts ${PROJECT_SOURCE_DIR}/simplenet.py
                    SimpleNet
                    ${PROJECT_SOURCE_DIR}/saved_simplenet_model_hip.pt
                    --input_tensor_file
                    ${PROJECT_SOURCE_DIR}/saved_simplenet_input_tensor_hip.pt
                    --output_model_file
                    ${PROJECT_BINARY_DIR}/saved_simplenet_model_hip.pt
                    --test)

    # 3b. Check the model can be loaded from file and run on two HIP devices in
    # Python and that its outputs meet expectations
    add_test(
      NAME multigpu_infer_python
      COMMAND ${Python_EXECUTABLE}
              ${PROJECT_SOURCE_DIR}/multigpu_infer_python.py --device_type hip
              --filepath ${PROJECT_BINARY_DIR}
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    # 4b. Check the model can be loaded from file and run on two HIP devices in
    # Fortran and that its outputs meet expectations
    add_test(
      NAME multigpu_infer_fortran
      COMMAND multigpu_infer_fortran hip
              ${PROJECT_BINARY_DIR}/saved_multigpu_model_hip.pt
      # Command line arguments for device type and model file
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    set_tests_properties(
      multigpu_infer_fortran PROPERTIES PASS_REGULAR_EXPRESSION
                                        "MultiGPU example ran successfully")
  endif()

  if("${GPU_DEVICE}" STREQUAL "XPU")
    # 1c. Write the PyTorch model out with the XPU device type and check it
    #     runs on a XPU device and its outputs meet expectations
    add_test(NAME example_multigpu_simplenet
             COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/simplenet.py
                     --device_type xpu)

    # 2c. Run the pt2ts script to convert the model file to TorchScript format
    #     and run basic checks
    add_test(
      NAME example_multigpu_pt2ts
      COMMAND pt2ts ${PROJECT_SOURCE_DIR}/simplenet.py
                    SimpleNet
                    ${PROJECT_SOURCE_DIR}/saved_simplenet_model_xpu.pt
                    --input_tensor_file
                    ${PROJECT_SOURCE_DIR}/saved_simplenet_input_tensor_xpu.pt
                    --output_model_file
                    ${PROJECT_BINARY_DIR}/saved_simplenet_model_xpu.pt
                    --test)

    # 3c. Check the model can be loaded from file and run on two XPU devices in
    # Python and that its outputs meet expectations
    add_test(
      NAME example_multigpu_infer_python
      COMMAND ${Python_EXECUTABLE}
              ${PROJECT_SOURCE_DIR}/multigpu_infer_python.py --device_type xpu
              --filepath ${PROJECT_BINARY_DIR}
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    # 4c. Check the model can be loaded from file and run on two XPU devices in
    # Fortran and that its outputs meet expectations
    add_test(
      NAME example_multigpu_infer_fortran
      COMMAND multigpu_infer_fortran xpu
              ${PROJECT_BINARY_DIR}/saved_multigpu_model_xpu.pt
      # Command line arguments for device type and model file
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    set_tests_properties(
      example_multigpu_infer_fortran PROPERTIES PASS_REGULAR_EXPRESSION
      "MultiGPU example ran successfully")
  endif()

  if("${GPU_DEVICE}" STREQUAL "MPS")
    # 1d. Write the PyTorch model out with the MPS device type and check it
    #     runs on a MPS device and its outputs meet expectations
    add_test(NAME example_multigpu_simplenet
             COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/simplenet.py
                     --device_type mps)

    # 2d. Run the pt2ts script to convert the model file to TorchScript format
    #     and run basic checks
    add_test(
      NAME example_multigpu_pt2ts
      COMMAND pt2ts ${PROJECT_SOURCE_DIR}/simplenet.py
                    SimpleNet
                    ${PROJECT_SOURCE_DIR}/saved_simplenet_model_mps.pt
                    --input_tensor_file
                    ${PROJECT_SOURCE_DIR}/saved_simplenet_input_tensor_mps.pt
                    --output_model_file
                    ${PROJECT_BINARY_DIR}/saved_simplenet_model_mps.pt
                    --test)

    # 3d. Check the model can be loaded from file and run on one MPS device in
    # Python and that its outputs meet expectations
    add_test(
      NAME example_multigpu_infer_python
      COMMAND ${Python_EXECUTABLE}
              ${PROJECT_SOURCE_DIR}/multigpu_infer_python.py --device_type mps
              --filepath ${PROJECT_BINARY_DIR}
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    # 4d. Check the model can be loaded from file and run on one MPS device in
    # Fortran and that its outputs meet expectations
    add_test(
      NAME example_multigpu_infer_fortran
      COMMAND multigpu_infer_fortran mps
              ${PROJECT_BINARY_DIR}/saved_multigpu_model_mps.pt
      # Command line arguments for device type and model file
      WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    set_tests_properties(
      example_multigpu_infer_fortran PROPERTIES PASS_REGULAR_EXPRESSION
      "MultiGPU example ran successfully")
  endif()
endif()
