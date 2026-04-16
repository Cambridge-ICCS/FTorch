cmake_minimum_required(VERSION 3.15...3.31)
# policy CMP0076 - target_sources source files are relative to file where
# target_sources is run
cmake_policy(SET CMP0076 NEW)

set(PROJECT_NAME fno1dExample)

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
add_executable(fno1d_infer_fortran fno1d_infer_fortran.f90)
target_link_libraries(fno1d_infer_fortran PRIVATE FTorch::ftorch)

# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)

  # 1. Check the PyTorch model runs and its outputs meet expectations
  add_test(NAME fno1d COMMAND ${Python_EXECUTABLE}
                                  ${PROJECT_SOURCE_DIR}/fno1d.py)

  # 2. Check the model is saved to file in the expected location with the
  #   pt2ts.py script
  add_test(
    NAME pt2ts
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/pt2ts.py --filepath
            ${PROJECT_BINARY_DIR}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

  # 3. Check the model can be loaded from file and run in Python and that its
  #   outputs meet expectations
  add_test(
    NAME fno1d_infer_python
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/fno1d_infer_python.py
            --filepath ${PROJECT_BINARY_DIR}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

  # 4. Check the model can be loaded from file and run in Fortran and that its
  #   outputs meet expectations
  add_test(
    NAME fno1d_infer_fortran
    COMMAND
      fno1d_infer_fortran ${PROJECT_BINARY_DIR}/fno1d_sine.pt
      # Command line argument: model file
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(
    fno1d_infer_fortran PROPERTIES PASS_REGULAR_EXPRESSION
    "fno1d example ran successfully")
endif()
