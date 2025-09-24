cmake_minimum_required(VERSION 3.15...3.31)
# policy CMP0076 - target_sources source files are relative to file where
# target_sources is run
cmake_policy(SET CMP0076 NEW)

set(PROJECT_NAME TensorExample)

project(${PROJECT_NAME} LANGUAGES Fortran)

# Build in Debug mode if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "" FORCE)
endif()

find_package(FTorch REQUIRED)
message(STATUS "Building with Fortran PyTorch coupling")

# Fortran example
add_executable(tensor_manipulation tensor_manipulation.f90)
target_link_libraries(tensor_manipulation PRIVATE FTorch::ftorch)

# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)

  # 1. Check the Fortran example runs and its outputs meet expectations
  add_test(
    NAME example1_tensor_manipulation
    COMMAND tensor_manipulation
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(example1_tensor_manipulation
                       PROPERTIES PASS_REGULAR_EXPRESSION
                       "Tensor manipulation example ran successfully")
endif()
