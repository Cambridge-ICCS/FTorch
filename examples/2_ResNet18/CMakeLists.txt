cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
#policy CMP0076 - target_sources source files are relative to file where target_sources is run
cmake_policy (SET CMP0076 NEW)

set(PROJECT_NAME ResNetExample)

project(${PROJECT_NAME} LANGUAGES Fortran)

# Build in Debug mode if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

find_package(FTorch)
message(STATUS "Building with Fortran PyTorch coupling")

# Fortran example
add_executable(resnet_infer_fortran resnet_infer_fortran.f90)
target_link_libraries(resnet_infer_fortran PRIVATE FTorch::ftorch)

# Integration testing
include(CTest)
add_test(NAME resnet18
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/resnet18.py
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set_tests_properties(resnet18 PROPERTIES PASS_REGULAR_EXPRESSION "0.8846225142478943")
set_tests_properties(resnet18 PROPERTIES PASS_REGULAR_EXPRESSION "0.045805174857378006")
set_tests_properties(resnet18 PROPERTIES PASS_REGULAR_EXPRESSION "0.0442761555314064")
set_tests_properties(resnet18 PROPERTIES PASS_REGULAR_EXPRESSION "0.005621383432298899")
set_tests_properties(resnet18 PROPERTIES PASS_REGULAR_EXPRESSION "0.004652013536542654")
add_test(NAME pt2ts
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pt2ts.py
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
add_test(NAME resnet_infer_fortran
  COMMAND resnet_infer_fortran ../saved_resnet18_model_cpu.pt)
set_tests_properties(resnet_infer_fortran PROPERTIES PASS_REGULAR_EXPRESSION "0.884623706")
