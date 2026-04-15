if(CMAKE_BUILD_TESTS)

  # Function for preprocessing examples
  function(add_example subdir test_basename)
    set(sourcedir ${CMAKE_CURRENT_SOURCE_DIR}/../test/integration/${subdir})
    set(exampledir ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
    add_custom_command(
      OUTPUT ${exampledir}/${test_basename}.f90
      COMMAND fypp ${sourcedir}/${test_basename}.fypp
                   ${exampledir}/${test_basename}.f90
      DEPENDS ${sourcedir}/${test_basename}.fypp
    )
    add_custom_target(generate_${test_basename}_example
      DEPENDS ${examledir}/${test_basename}.f90
    )
  endfunction()

  # Preprocessing for examples
  add_example(01_Tensor tensor_manipulation)
  # add_example(02_SimpleNet simplenet_infer_fortran)
  # add_example(03_ResNet resnet_infer_fortran)
  # add_example(04_Batching batching_demo_fortran)
  # TODO: more

  # Build examples
  add_subdirectory(01_Tensor)
  # add_subdirectory(02_SimpleNet)
  # add_subdirectory(03_ResNet)
  # add_subdirectory(04_Batching)
  # add_subdirectory(05_MultiIO)
  # add_subdirectory(06_Looping)
  # # NOTE: Example 06_Looping is built here but not run as part of the
  # #       integration test suite. This is because it demonstrates
  # #       'good' versus 'bad' practice, as opposed to functionality.
  # if(NOT "${GPU_DEVICE}" STREQUAL "NONE" AND MULTI_GPU)
  #   add_subdirectory(07_MultiGPU)
  # else()
  #   message(STATUS "Skipping Multi GPU integration tests")
  # endif()
  # # NOTE: We do not currently support running with MPI on Windows
  # if(UNIX)
  #   add_subdirectory(08_MPI)
  # endif()
  # add_subdirectory(09_Autograd)
  # add_subdirectory(10_Optimizers)
endif()
