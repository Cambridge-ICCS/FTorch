cmake_minimum_required(VERSION 3.15...3.31)
cmake_policy (SET CMP0076 NEW)

project("FTorch unit tests" VERSION 1.0.0 LANGUAGES Fortran)

find_package(FTorch)
message(STATUS "Building with Fortran PyTorch coupling")

find_package(PFUNIT REQUIRED)

add_pfunit_ctest(
  test_tensor_constructors_destructors TEST_SOURCES
  test_tensor_constructors_destructors.pf LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(test_tensor_interrogation
  TEST_SOURCES test_tensor_interrogation.pf LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(test_operator_overloads
  TEST_SOURCES test_tensor_operator_overloads.pf LINK_LIBRARIES FTorch::ftorch)

if("${GPU_DEVICE}" STREQUAL "CUDA")
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(ERROR "No CUDA support")
  endif()
  add_pfunit_ctest(test_tensor_interrogation_cuda
    TEST_SOURCES test_tensor_interrogation_cuda.pf
    LINK_LIBRARIES FTorch::ftorch)
endif()
