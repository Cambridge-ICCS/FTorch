cmake_minimum_required(VERSION 3.15...3.31)
# policy CMP0076 - target_sources source files are relative to file where
# target_sources is run
cmake_policy(SET CMP0076 NEW)

set(PROJECT_NAME ReqGradBugMWE)

project(${PROJECT_NAME} LANGUAGES Fortran)

# Build in Debug mode if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "" FORCE)
endif()

find_package(FTorch)
message(STATUS "Building with Fortran PyTorch coupling")

# Fortran example
add_executable(rgmwe rgmwe.f90)
target_link_libraries(rgmwe PRIVATE FTorch::ftorch)

# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)

  # Check the Fortran rgmwe script runs successfully
  add_test(
    NAME rgmwe
    COMMAND rgmwe
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(rgmwe PROPERTIES PASS_REGULAR_EXPRESSION
                                         "rgmwe example ran successfully")
endif()
