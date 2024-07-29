find_package(FTorch)
find_package(MPI REQUIRED)
message(STATUS "Building with Fortran PyTorch coupling")

# Include PFUNIT directories
include_directories(${PFUNIT}/include)

set(pf_tests
  test_constructors.pf)

add_pfunit_tests(new_tests ${pf_tests})
add_library(new_pftests EXCLUDE_FROM_ALL ${pf_tests})
target_link_libraries(new_ptests PRIVATE FTorch::ftorch MPI::MPI_Fortran ${PFUNIT}/lib/libpfunit.a)


# Integration testing
if(CMAKE_BUILD_TESTS)
  include(CTest)
  add_test(NAME constructors COMMAND constructors WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
endif()
