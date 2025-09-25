cmake_minimum_required(VERSION 3.15...3.31)
cmake_policy(SET CMP0076 NEW)

project(
  "FTorch unit tests"
  VERSION 1.0.0
  LANGUAGES Fortran)

if(CMAKE_BUILD_TESTS)
  find_package(FTorch QUIET)
else()
  find_package(FTorch REQUIRED)
endif()
message(STATUS "Building with Fortran PyTorch coupling")

find_package(PFUNIT REQUIRED)

add_pfunit_ctest(
  unittest_tensor_constructors_destructors TEST_SOURCES
  unittest_tensor_constructors_destructors.pf LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(unittest_tensor_interrogation
  TEST_SOURCES unittest_tensor_interrogation.pf LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(unittest_tensor_manipulation
  TEST_SOURCES unittest_tensor_manipulation.pf LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(
  unittest_operator_overloads TEST_SOURCES unittest_tensor_operator_overloads.pf
  LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(test_operators
  TEST_SOURCES test_tensor_operators.pf  LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(test_operators_autograd
  TEST_SOURCES test_tensor_operators_autograd.pf  LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(
  unittest_operator_overloads_autograd TEST_SOURCES
  unittest_tensor_operator_overloads_autograd.pf LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(unittest_tensor_autograd
  TEST_SOURCES unittest_tensor_autograd.pf LINK_LIBRARIES FTorch::ftorch)

if("${GPU_DEVICE}" STREQUAL "CUDA" OR "${GPU_DEVICE}" STREQUAL "HIP")
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
  add_pfunit_ctest(
    unittest_tensor_interrogation_cuda TEST_SOURCES
    unittest_tensor_interrogation_cuda.pf LINK_LIBRARIES FTorch::ftorch)
  add_pfunit_ctest(
    test_tensor_manipulation_cuda TEST_SOURCES
    test_tensor_manipulation_cuda.pf LINK_LIBRARIES FTorch::ftorch)
endif()
