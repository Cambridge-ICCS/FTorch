if(CMAKE_BUILD_TESTS)
  add_subdirectory(1_Tensor)
  add_subdirectory(2_SimpleNet)
  add_subdirectory(3_ResNet)
  add_subdirectory(4_MultiIO)
  add_subdirectory(5_Looping)
  # NOTE: Example 5_Looping is built here but not run as part of the integration
  #       test suite. This is because it demonstrates 'good' versus 'bad'
  #       practice, as opposed to functionality.
  if(NOT "${GPU_DEVICE}" STREQUAL "NONE" AND MULTI_GPU)
    add_subdirectory(6_MultiGPU)
  endif()
  # NOTE: We do not currently support running with MPI on Windows
  if(UNIX)
    add_subdirectory(7_MPI)
  endif()
  add_subdirectory(8_Autograd)
endif()
