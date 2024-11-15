cmake_minimum_required(VERSION 3.12)
cmake_policy (SET CMP0076 NEW)

project("FTorch unit tests" VERSION 1.0.0 LANGUAGES Fortran)

find_package(FTorch REQUIRED)
message(STATUS "Building with Fortran PyTorch coupling")

find_package(PFUNIT REQUIRED)

add_pfunit_ctest(test_ftorch TEST_SOURCES test_ftorch.pf LINK_LIBRARIES FTorch::ftorch)
