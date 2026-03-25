cmake_minimum_required(VERSION 3.18...3.31)
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

if(CMAKE_BUILD_TESTS)
  find_package(FTorch QUIET)
else()
  find_package(FTorch REQUIRED)
endif()
message(STATUS "Building with Fortran PyTorch coupling")

# Install Python dependencies
find_package(Python COMPONENTS Interpreter REQUIRED)
if(NOT DEFINED ENV{VIRTUAL_ENV} AND NOT DEFINED ENV{CONDA_PREFIX})
  message(FATAL_ERROR "Please activate your virtualenv or conda environment")
endif()
execute_process(COMMAND ${Python_EXECUTABLE} -m pip install -r
                        ${PROJECT_SOURCE_DIR}/requirements.txt)

# Fortran examples
add_executable(tensor_manipulation_backward tensor_manipulation_backward.f90)
target_link_libraries(tensor_manipulation_backward PRIVATE FTorch::ftorch)
add_executable(simplenet_backward simplenet_backward.f90)
target_link_libraries(simplenet_backward PRIVATE FTorch::ftorch)

# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)

  # 1. Check the Python tensor_manipulation_backward script runs successfully
  add_test(NAME example_autograd_tensor_manipulation_backward_python
           COMMAND ${Python_EXECUTABLE}
                   ${PROJECT_SOURCE_DIR}/tensor_manipulation_backward.py)

  # 2. Check the Fortran tensor_manipulation_backward script runs successfully
  add_test(
    NAME example_autograd_tensor_manipulation_backward_fortran
    COMMAND tensor_manipulation_backward
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(example_autograd_tensor_manipulation_backward_fortran
                       PROPERTIES PASS_REGULAR_EXPRESSION
                       "Tensor manipulation Autograd example ran successfully")

  # 3. Check the Python simplenet_backward Autograd script runs successfully
  add_test(NAME example_autograd_simplenet_backward_python
           COMMAND ${Python_EXECUTABLE}
                   ${PROJECT_SOURCE_DIR}/simplenet_backward.py)

  # 4. Check the Fortran simplenet_backward Autograd script runs successfully
  add_test(
    NAME example_autograd_simplenet_backward_fortran
    COMMAND simplenet_backward
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(example_autograd_simplenet_backward_fortran
                       PROPERTIES PASS_REGULAR_EXPRESSION
                       "SimpleNet Autograd example ran successfully")
endif()
