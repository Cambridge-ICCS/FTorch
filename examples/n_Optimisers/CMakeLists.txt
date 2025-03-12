cmake_minimum_required(VERSION 3.15...3.31)
# policy CMP0076 - target_sources source files are relative to file where
# target_sources is run
cmake_policy(SET CMP0076 NEW)

set(PROJECT_NAME OptimisersExample)

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
add_executable(optimisers optimisers.f90)
target_link_libraries(optimisers PRIVATE FTorch::ftorch)

# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)

  # 1. Check the Python Optimisers script runs successfully
  add_test(NAME pyoptim COMMAND ${Python_EXECUTABLE}
                                   ${PROJECT_SOURCE_DIR}/optimisers.py)

  # 2. Check the Fortran Optimisers script runs successfully
  add_test(
    NAME foptim
    COMMAND optimisers
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(foptim PROPERTIES PASS_REGULAR_EXPRESSION
                                         "Optimisers example ran successfully")
endif()
