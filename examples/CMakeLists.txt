if(CMAKE_BUILD_TESTS)
  add_subdirectory(1_SimpleNet)
  add_subdirectory(2_ResNet18)
  if(NOT "${GPU_DEVICE}" STREQUAL "NONE")
    add_subdirectory(3_MultiGPU)
  endif()
  add_subdirectory(4_MultiIO)
  # NOTE: Example 5_Looping is built here but not run as part of the integration
  #       test suite. This is because it demonstrates 'good' versus 'bad'
  #       practice, as opposed to functionality.
  add_subdirectory(5_Looping)
  add_subdirectory(6_Autograd)
endif()
