cmake_minimum_required(VERSION 3.15...3.31)
# policy CMP0076 - target_sources source files are relative to file where
# target_sources is run
cmake_policy(SET CMP0076 NEW)

set(PROJECT_NAME MultiGPUExample)

project(${PROJECT_NAME} LANGUAGES Fortran)

# Build in Debug mode if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "" FORCE)
endif()

find_package(FTorch)
find_package(MPI REQUIRED)
message(STATUS "Building with Fortran PyTorch coupling")

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(WARNING "No CUDA support")
endif()

# Fortran example
add_executable(multigpu_infer_fortran multigpu_infer_fortran.f90)
target_link_libraries(multigpu_infer_fortran PRIVATE FTorch::ftorch)
target_link_libraries(multigpu_infer_fortran PRIVATE MPI::MPI_Fortran)

# Integration testing
if (CMAKE_BUILD_TESTS)
  include(CTest)

  # 1. Check the PyTorch model runs and its outputs meet expectations
  add_test(NAME multigpu COMMAND ${Python_EXECUTABLE}
                                  ${PROJECT_SOURCE_DIR}/multigpu.py)

  # 2. Check the model is saved to file in the expected location with the
  #   pt2ts.py script
  add_test(
    NAME pt2ts
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/pt2ts.py
            ${PROJECT_BINARY_DIR} # Command line argument: filepath for saving
                                  # the model
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

  # 3. Check the model can be loaded from file and run in Python and that its
  #   outputs meet expectations
  add_test(
    NAME multigpu_infer_python
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/multigpu_infer_python.py
            ${PROJECT_BINARY_DIR} # Command line argument: filepath to find the
                                  # model
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

  # 4. Check the model can be loaded from file and run in Fortran and that its
  #   outputs meet expectations
  add_test(
    NAME multigpu_infer_fortran
    COMMAND
      multigpu_infer_fortran ${PROJECT_BINARY_DIR}/saved_multigpu_model_cpu.pt
      # Command line argument: model file
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(
    multigpu_infer_fortran PROPERTIES PASS_REGULAR_EXPRESSION
    "MultiGPU example ran successfully")
endif()
