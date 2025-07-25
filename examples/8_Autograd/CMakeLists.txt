cmake_minimum_required(VERSION 3.15...3.31)
# policy CMP0076 - target_sources source files are relative to file where
# target_sources is run
cmake_policy(SET CMP0076 NEW)

set(PROJECT_NAME AutogradExample)

project(${PROJECT_NAME} LANGUAGES Fortran)

# Build in Debug mode if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "" FORCE)
endif()

find_package(FTorch)
message(STATUS "Building with Fortran PyTorch coupling")

# Install Python dependencies
find_package(Python COMPONENTS Interpreter REQUIRED)
if(NOT DEFINED ENV{VIRTUAL_ENV} OR DEFINED ENV{CONDA_PREFIX})
  message(FATAL_ERROR "Please activate your virtualenv or conda environment")
endif()
execute_process(COMMAND ${Python_EXECUTABLE} -m pip install -r
                        ${PROJECT_SOURCE_DIR}/requirements.txt)

# Fortran example
add_executable(autograd autograd.f90)
target_link_libraries(autograd PRIVATE FTorch::ftorch)

# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)

  # 1. Check the Python Autograd script runs successfully
  add_test(NAME example8_pyautograd COMMAND ${Python_EXECUTABLE}
                                            ${PROJECT_SOURCE_DIR}/autograd.py)

  # 2. Check the Fortran Autograd script runs successfully
  add_test(
    NAME example8_fautograd
    COMMAND autograd
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(example8_fautograd PROPERTIES PASS_REGULAR_EXPRESSION
                       "Autograd example ran successfully")
endif()
