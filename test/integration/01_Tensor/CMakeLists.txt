cmake_minimum_required(VERSION 3.18...3.31)
# policy CMP0076 - target_sources source files are relative to file where
# target_sources is run
cmake_policy(SET CMP0076 NEW)

if ("${TEST}" STREQUAL "NONE")
  set(prefix "example")
  set(name "example")
else()
  set(prefix "integrationtest")
  set(name "integration test")
endif()

set(PROJECT_NAME TensorIntegrationTest)

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

# Fortran integration test
add_executable(${prefix}_tensor_manipulation tensor_manipulation.f90)
target_link_libraries(${prefix}_tensor_manipulation PRIVATE FTorch::ftorch)

# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)

  # 1. Check the Fortran integration test runs and its outputs meet expectations
  add_test(
    NAME ${prefix}_tensor_manipulation
    COMMAND ${prefix}_tensor_manipulation
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(${prefix}_tensor_manipulation
                       PROPERTIES PASS_REGULAR_EXPRESSION
                       "Tensor manipulation ${name} ran successfully")
endif()
