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
  if(NOT ("${GPU_DEVICE}" STREQUAL "NONE" OR "${GPU_DEVICE}" STREQUAL ""))
    add_subdirectory(07_MultiGPU)
  else()
    message(STATUS "Skipping Multi GPU integration tests")
  endif()
  # NOTE: We do not currently support running with MPI on Windows
  if(UNIX)
    find_package(MPI COMPONENTS Fortran QUIET)
    if(MPI_Fortran_FOUND)
      add_subdirectory(08_MPI)
    else()
      message(STATUS "Skipping MPI integration tests (MPI Fortran not found)")
    endif()
  endif()
  add_subdirectory(09_Autograd)
  add_subdirectory(10_Optimizers)
  add_subdirectory(11_Training)
  add_subdirectory(12_Permutation)
  # NOTE: This example is built only when CUDA support is enabled since it
  #       uses CUDA Fortran to allocate arrays on the GPU and requires a
  #       CUDA Fortran capable compiler (e.g. nvfortran)
  if("${GPU_DEVICE}" STREQUAL "CUDA" AND CMAKE_Fortran_COMPILER_ID MATCHES
                                         "NVHPC|PGI")
    add_subdirectory(13_GPUPointer)
  else()
    message(
      STATUS
        "Skipping GPU pointer integration tests (requires CUDA support and nvfortran)"
    )
  endif()
endif()
