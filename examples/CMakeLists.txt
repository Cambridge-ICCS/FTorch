if(CMAKE_BUILD_TESTS)
  add_subdirectory(1_SimpleNet)
  add_subdirectory(2_ResNet18)
  add_subdirectory(3_MultiIO)
  add_subdirectory(4_Looping)
  # NOTE: Example 4_Looping is built here but not run as part of the integration
  #       test suite. This is because it demonstrates 'good' versus 'bad'
  #       practice, as opposed to functionality.
  if(NOT "${GPU_DEVICE}" STREQUAL "NONE")
    add_subdirectory(5_MultiGPU)
  endif()
  # NOTE: We do not currently support running with MPI on Windows
  if(UNIX)
    add_subdirectory(6_MPI)
  endif()
  add_subdirectory(7_Autograd)
endif()
