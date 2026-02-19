cmake_minimum_required(VERSION 3.18...3.31)
# policy CMP0076 - target_sources source files are relative to file where
# target_sources is run
cmake_policy(SET CMP0076 NEW)

set(PROJECT_NAME TensorExample)

project(${PROJECT_NAME} LANGUAGES C CXX Fortran)

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

# Fortran example
add_executable(tensor_manipulation tensor_manipulation.f90)
target_link_libraries(tensor_manipulation PRIVATE FTorch::ftorch)

# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)

  # 1. Check the Fortran example runs and its outputs meet expectations
  add_test(
    NAME example_tensor_manipulation
    COMMAND tensor_manipulation
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(example_tensor_manipulation
                       PROPERTIES PASS_REGULAR_EXPRESSION
                       "Tensor manipulation example ran successfully")
endif()
