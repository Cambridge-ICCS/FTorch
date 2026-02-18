if(CMAKE_BUILD_TESTS)
  add_subdirectory(1_Tensor)
  add_subdirectory(2_SimpleNet)
  add_subdirectory(3_ResNet)
  add_subdirectory(4_Batching)
  add_subdirectory(5_MultiIO)
  add_subdirectory(6_Looping)
  # NOTE: Example 6_Looping is built here but not run as part of the integration
  #       test suite. This is because it demonstrates 'good' versus 'bad'
  #       practice, as opposed to functionality.
  if(NOT "${GPU_DEVICE}" STREQUAL "NONE")
    add_subdirectory(7_MultiGPU)
  else()
    message(STATUS "Skipping Multi GPU integration tests")
  endif()
  # NOTE: We do not currently support running with MPI on Windows
  if(UNIX)
    add_subdirectory(8_MPI)
  endif()
  add_subdirectory(9_Autograd)
endif()
