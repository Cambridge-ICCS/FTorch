cmake_minimum_required(VERSION 3.15...3.31)
# policy CMP0076 - target_sources source files are relative to file where
# target_sources is run
cmake_policy(SET CMP0076 NEW)

set(PROJECT_NAME LoopingExample)

project(${PROJECT_NAME} LANGUAGES Fortran)

# Build in Debug mode if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "" FORCE)
endif()

find_package(FTorch REQUIRED)
message(STATUS "Building with Fortran PyTorch coupling")

# Install Python dependencies
find_package(Python COMPONENTS Interpreter REQUIRED)
if(NOT DEFINED ENV{VIRTUAL_ENV} AND NOT DEFINED ENV{CONDA_PREFIX})
  message(FATAL_ERROR "Please activate your virtualenv or conda environment")
endif()
execute_process(COMMAND ${Python_EXECUTABLE} -m pip install -r
                        ${PROJECT_SOURCE_DIR}/requirements.txt)

# Fortran example - bad
add_executable(example5_simplenet_infer_fortran_bad
               bad/simplenet_infer_fortran.f90)
target_link_libraries(example5_simplenet_infer_fortran_bad
                      PRIVATE FTorch::ftorch)
target_sources(example5_simplenet_infer_fortran_bad
               PRIVATE bad/fortran_ml_mod.f90)

# Fortran example - good
add_executable(example5_simplenet_infer_fortran_good
               good/simplenet_infer_fortran.f90)
target_link_libraries(example5_simplenet_infer_fortran_good
                      PRIVATE FTorch::ftorch)
target_sources(example5_simplenet_infer_fortran_good
               PRIVATE good/fortran_ml_mod.f90)
