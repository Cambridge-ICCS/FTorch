cmake_minimum_required(VERSION 3.15...3.31)
# policy CMP0076 - target_sources source files are relative to file where
# target_sources is run
cmake_policy(SET CMP0076 NEW)

set(PROJECT_NAME MPIExample)

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
find_package(MPI REQUIRED)
message(STATUS "Building with Fortran PyTorch coupling")

# Install Python dependencies
find_package(Python COMPONENTS Interpreter REQUIRED)
if(NOT DEFINED ENV{VIRTUAL_ENV} AND NOT DEFINED ENV{CONDA_PREFIX})
  message(FATAL_ERROR "Please activate your virtualenv or conda environment")
endif()
execute_process(COMMAND ${Python_EXECUTABLE} -m pip install -r
                        ${PROJECT_SOURCE_DIR}/requirements.txt)

# Fortran example
add_executable(mpi_infer_fortran mpi_infer_fortran.f90)
target_link_libraries(mpi_infer_fortran PRIVATE FTorch::ftorch)
target_link_libraries(mpi_infer_fortran PRIVATE MPI::MPI_Fortran)

# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)

  # 1. Check the PyTorch model runs and its outputs meet expectations
  add_test(NAME example7_simplenet COMMAND ${Python_EXECUTABLE}
                                           ${PROJECT_SOURCE_DIR}/simplenet.py)

  # 2. Check the model is saved to file in the expected location with the
  #   pt2ts.py script
  add_test(
    NAME example7_pt2ts
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/pt2ts.py --filepath
            ${PROJECT_BINARY_DIR}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

  # 3. Check the model can be loaded from file and run with MPI in Python and
  #   that its outputs meet expectations
  add_test(
    NAME example7_mpi_infer_python
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${Python_EXECUTABLE}
      ${PROJECT_SOURCE_DIR}/mpi_infer_python.py --filepath ${PROJECT_BINARY_DIR}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(
    example7_mpi_infer_python PROPERTIES PASS_REGULAR_EXPRESSION
    "MPI Python example ran successfully")

  # 4. Check the model can be loaded from file and run with MPI in Fortran and
  #   that its outputs meet expectations
  add_test(
    NAME example7_mpi_infer_fortran
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ./mpi_infer_fortran
      ${PROJECT_BINARY_DIR}/saved_simplenet_model_cpu.pt
      # Command line argument: model file
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
  set_tests_properties(
    example7_mpi_infer_fortran PROPERTIES PASS_REGULAR_EXPRESSION
    "MPI Fortran example ran successfully")
endif()
