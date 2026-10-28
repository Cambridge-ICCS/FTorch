cmake_minimum_required(VERSION 3.18...3.31)
# policy CMP0076 - target_sources source files are relative to file where
# target_sources is run
cmake_policy(SET CMP0076 NEW)

set(PROJECT_NAME GPUPointerExample)

project(${PROJECT_NAME} LANGUAGES Fortran)

# Build in Debug mode if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "" FORCE)
endif()

# This example uses CUDA Fortran to define arrays on the GPU and requires an
# NVIDIA GPU, a CUDA Fortran capable compiler (e.g. nvfortran), and an FTorch
# installation built with -DGPU_DEVICE=CUDA
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  set(CUDA_FORTRAN_FLAG "-cuda")
else()
  message(
    WARNING
      "The GPUPointer example uses CUDA Fortran and has only been tested with "
      "a CUDA Fortran capable compiler (e.g. nvfortran). Compilation with "
      "${CMAKE_Fortran_COMPILER_ID} is likely to fail."
  )
endif()

if(CMAKE_BUILD_TESTS)
  find_package(FTorch QUIET)
else()
  find_package(FTorch REQUIRED)
endif()
message(STATUS "Building with Fortran PyTorch coupling")

# Fortran example
add_executable(gpu_tensor_pointer gpu_tensor_pointer.f90)
target_compile_options(gpu_tensor_pointer PRIVATE ${CUDA_FORTRAN_FLAG})
target_link_libraries(gpu_tensor_pointer PRIVATE FTorch::ftorch)

# Integration testing
# NOTE: This test is only registered when GPU_DEVICE=CUDA is explicitly passed,
# since running the example requires a CUDA-capable GPU
if(CMAKE_BUILD_TESTS AND "${GPU_DEVICE}" STREQUAL "CUDA")
  include(CTest)

  add_test(
    NAME example_gpu_tensor_pointer
    COMMAND gpu_tensor_pointer
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(example_gpu_tensor_pointer
                       PROPERTIES PASS_REGULAR_EXPRESSION
                       "GPU device pointer example ran successfully")
endif()
