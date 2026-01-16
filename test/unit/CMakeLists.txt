cmake_minimum_required(VERSION 3.15...3.31)
cmake_policy(SET CMP0076 NEW)
project("FTorch unit tests" VERSION 1.0.0 LANGUAGES Fortran)

if(CMAKE_BUILD_TESTS)
  find_package(FTorch QUIET)
else()
  find_package(FTorch REQUIRED)
endif()
message(STATUS "Building FTorch unit tests")

# Find the veggies package
if(NOT TARGET "veggies")
  find_package("veggies" REQUIRED)
endif()

# Define CPU-based unit tests
set(
  TESTS
  # TODO: "tensor_constructors_destructors"
  # TODO: "tensor_interrogation"
  # TODO: "tensor_manipulation"
  "tensor_operators"
  # TODO: "tensor_operator_overloads"
  # TODO: "tensor_operator_overloads_autograd"
  # TODO: "tensor_operators_autograd"
)

# Define GPU-based unit tests
if("${GPU_DEVICE}" STREQUAL "CUDA" OR "${GPU_DEVICE}" STREQUAL "HIP")
  if("${GPU_DEVICE}" STREQUAL "CUDA")
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      enable_language(CUDA)
    else()
      message(ERROR "No CUDA support")
    endif()
  endif()
  if("${GPU_DEVICE}" STREQUAL "HIP")
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
      enable_language(HIP)
    else()
      message(WARNING "No HIP support")
    endif()
  endif()
  # set(
  #   TESTS
  #   ${TESTS}
  #   # TODO: "tensor_interrogation_cuda"
  #   # TODO: "tensor_manipulation_cuda"
  # )
endif()

# Set the test driver
set(TEST_SOURCES "main.f90")

# Accumulate the unit tests then create an executable
foreach(test IN LISTS TESTS)
  string(MAKE_C_IDENTIFIER ${test} test)
  list(APPEND TEST_SOURCES "unittest_${test}.f90")
endforeach()
add_executable("FTorch-unit-tests" "${TEST_SOURCES}")
target_link_libraries("FTorch-unit-tests"
                      PRIVATE FTorch::ftorch veggies)

# Register the unit tests with CTest
foreach(test IN LISTS TESTS)
  add_test("FTorch-unit-tests/${test}" "FTorch-unit-tests")
endforeach()
