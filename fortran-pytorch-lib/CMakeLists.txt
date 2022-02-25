cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(fortran_torch_bridge C CXX Fortran)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

# C++ executable
add_executable(ts_infer_cpp ts_inference.cpp)
target_link_libraries(ts_infer_cpp ${TORCH_LIBRARIES})

# C bindings and executable
add_executable(ts_infer_c ts_inference.c c_torch.cpp)
target_link_libraries(ts_infer_c ${TORCH_LIBRARIES})

add_library(ctorch SHARED c_torch.cpp)
set_target_properties(${LIB_NAME} PROPERTIES PUBLIC_HEADER "c_torch.h")
target_link_libraries(ctorch PRIVATE ${TORCH_LIBRARIES})

# Fortran module and executable
add_executable(ts_infer_fortran c_torch.cpp mod_torch.f90 ts_inference.f90)
target_link_libraries(ts_infer_fortran PRIVATE ${TORCH_LIBRARIES})
