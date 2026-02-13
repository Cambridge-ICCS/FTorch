# Torch Tensor API on CPU
add_pfunit_ctest(unittest_constructors_destructors
                 TEST_SOURCES unittest_constructors_destructors.pf
                 LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(unittest_interrogation
                 TEST_SOURCES unittest_interrogation.pf
                 LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(unittest_manipulation
                 TEST_SOURCES unittest_manipulation.pf
                 LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(unittest_operator_overloads
                 TEST_SOURCES unittest_operator_overloads.pf
                 LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(unittest_operators
                 TEST_SOURCES unittest_operators.pf
                 LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(unittest_operators_autograd
                 TEST_SOURCES unittest_operators_autograd.pf
                 LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(unittest_operator_overloads_autograd
                 TEST_SOURCES unittest_operator_overloads_autograd.pf
                 LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(unittest_autograd
                 TEST_SOURCES unittest_autograd.pf
                 LINK_LIBRARIES FTorch::ftorch)

# Unit tests on CUDA or HIP devices
if("${GPU_DEVICE}" STREQUAL "CUDA" OR "${GPU_DEVICE}" STREQUAL "HIP")
  if("${GPU_DEVICE}" STREQUAL "CUDA")
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      enable_language(CUDA)
    else()
      message(ERROR "No CUDA support")
    endif()
  endif()
  if("${GPU_DEVICE}" STREQUAL "HIP")
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
      enable_language(HIP)
    else()
      message(WARNING "No HIP support")
    endif()
  endif()
  add_pfunit_ctest(unittest_interrogation_cuda
                   TEST_SOURCES unittest_interrogation_cuda.pf
                   LINK_LIBRARIES FTorch::ftorch)
  add_pfunit_ctest(unittest_manipulation_cuda
                   TEST_SOURCES unittest_manipulation_cuda.pf
                   LINK_LIBRARIES FTorch::ftorch)
endif()
