if(CMAKE_BUILD_TESTS)
  add_subdirectory(01_Tensor)
  add_subdirectory(02_SimpleNet)
  add_subdirectory(03_ResNet)
  add_subdirectory(04_Batching)
  add_subdirectory(05_MultiIO)
  add_subdirectory(06_Looping)
  # NOTE: Example 06_Looping is built here but not run as part of the
  #       integration test suite. This is because it demonstrates
  #       'good' versus 'bad' practice, as opposed to functionality.
  if(NOT "${GPU_DEVICE}" STREQUAL "NONE")
    add_subdirectory(07_MultiGPU)
  else()
    message(STATUS "Skipping Multi GPU integration tests")
  endif()
  # NOTE: We do not currently support running with MPI on Windows
  if(UNIX)
    add_subdirectory(08_MPI)
  endif()
  add_subdirectory(09_Autograd)
  add_subdirectory(10_Optimizers)
endif()
