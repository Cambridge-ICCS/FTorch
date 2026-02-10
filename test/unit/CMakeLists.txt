cmake_minimum_required(VERSION 3.18...3.31)
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

# Set up fixtures for the unit tests
add_subdirectory(fixtures)

# Add units tests for the different modules
add_subdirectory(tensor)
add_subdirectory(model)
