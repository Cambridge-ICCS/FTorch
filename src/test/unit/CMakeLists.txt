cmake_minimum_required(VERSION 3.15...3.31 FATAL_ERROR)
cmake_policy (SET CMP0076 NEW)

project("FTorch unit tests" VERSION 1.0.0 LANGUAGES Fortran)

find_package(FTorch)
message(STATUS "Building with Fortran PyTorch coupling")

find_package(PFUNIT REQUIRED)

add_pfunit_ctest(test_constructors
  TEST_SOURCES test_constructors.pf LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(test_operator_overloads
  TEST_SOURCES test_tensor_operator_overloads.pf LINK_LIBRARIES FTorch::ftorch)
