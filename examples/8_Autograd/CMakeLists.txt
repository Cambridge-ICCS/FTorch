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
add_executable(autograd autograd.f90)
target_link_libraries(autograd PRIVATE FTorch::ftorch)
add_executable(simplenet simplenet.f90)
target_link_libraries(simplenet PRIVATE FTorch::ftorch)

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

  # 3. Check the Python SimpleNet Autograd script runs successfully
  add_test(NAME example8_pysimplenet COMMAND ${Python_EXECUTABLE}
                                             ${PROJECT_SOURCE_DIR}/simplenet.py)

  # 4. Check the Fortran SimpleNet Autograd script runs successfully
  add_test(
    NAME example8_fsimplenet
    COMMAND simplenet
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(example8_fsimplenet PROPERTIES PASS_REGULAR_EXPRESSION
                       "SimpleNet Autograd example ran successfully")
endif()
