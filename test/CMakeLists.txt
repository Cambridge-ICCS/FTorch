cmake_minimum_required(VERSION 3.18...3.31)
cmake_policy(SET CMP0076 NEW)

project(
  "FTorch test suite"
  VERSION 1.0.0
  LANGUAGES Fortran)

if(CMAKE_BUILD_TESTS)
  find_package(FTorch QUIET)
else()
  find_package(FTorch REQUIRED)
endif()
message(STATUS "Building with Fortran PyTorch coupling")

if(CMAKE_BUILD_TESTS)

  # # Unit tests
  # NOTE: We do not currently support unit testing on Windows
  # if(UNIX)
  #   add_subdirectory(unit)
  # endif()

  # Function for preprocessing integration tests
  function(add_integration_test subdir test_basename)
    set(workdir ${CMAKE_CURRENT_SOURCE_DIR}/integration/${subdir})
    add_custom_command(
      OUTPUT ${workdir}/${test_basename}.f90
      COMMAND fypp -DTEST=1 ${workdir}/${test_basename}.fypp
                   ${workdir}/${test_basename}.f90
      DEPENDS ${workdir}/${test_basename}.fypp
    )
    add_custom_target(generate_${test_basename}_integration_test
      DEPENDS ${workdir}/${test_basename}.f90
    )
    # TODO: Copy over CMakeLists and Python files as well
  endfunction()

  # Preprocessing for integration tests
  add_integration_test(01_Tensor tensor_manipulation)
  # TODO: more

  # Build integration tests
  add_subdirectory(integration)
endif()
