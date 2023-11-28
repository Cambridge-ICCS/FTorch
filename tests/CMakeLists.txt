cmake_minimum_required(VERSION 3.12)
cmake_policy (SET CMP0076 NEW)

project (PFUNIT_DEMO_BASIC
  VERSION 1.0.0
  LANGUAGES Fortran)

find_package(FTorch REQUIRED)
message(STATUS "Building with Fortran PyTorch coupling")

find_package(PFUNIT REQUIRED)
enable_testing()


# Add each test suite here with format:
# ADD_PFUNIT_TEST(test_name test_file extra_sources extra_sources_c)
ADD_PFUNIT_TEST (my_tests test_ftorch.pf "" "")

# Should work but doesn't
#target_include_directories(my_tests PRIVATE ${FTorch_INCLUDE_DIRS})
#TARGET_LINK_DIRECTORIES(my_tests PRIVATE ${FTorch_LIBRARY_DIRS})

# Link the libraries for FTorch
#target_link_libraries(my_tests PRIVATE ${FTorch_LIBRARIES})



# Temporary fix
target_include_directories(my_tests PRIVATE "/usr/local/include/ftorch/")
target_link_directories(my_tests PRIVATE "/usr/local/lib/")
target_link_directories(my_tests PRIVATE "/usr/local/include/ftorch/")

set(CMAKE_EXE_LINKER_FLAGS "-lftorch -L/usr/local/lib/ -L/usr/local/include/ftorch/")
# Cannot do this
#target_link_libraries(my_tests PRIVATE FTorch::ftorch)
