cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
set(PROJECT_NAME FTorch)
set(LIB_NAME ftorch)
set(PACKAGE_VERSION 0.1)

project(${PROJECT_NAME} VERSION ${PACKAGE_VERSION} LANGUAGES C CXX Fortran)

include(FortranCInterface)
FortranCInterface_VERIFY(CXX QUIET)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set RPATH behaviour
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
# Embed absolute paths to external libraries that are not part of
# the project, (they are expected to be at the same location on all
# machines the project will be deployed to
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Follow GNU conventions for installing directories
include(GNUInstallDirs)

# Define RPATH for executables via a relative expression to enable a
# fully relocatable package
file(RELATIVE_PATH relDir
 ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
 ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_RPATH $ORIGIN/${relDir})

# Dependencies
find_package(Torch REQUIRED)

# Library with C and Fortran bindings
add_library(${LIB_NAME} SHARED ctorch.cpp ftorch.f90)
add_library(${PROJECT_NAME}::${LIB_NAME} ALIAS ${LIB_NAME})
set_target_properties(${LIB_NAME} PROPERTIES
  PUBLIC_HEADER "ctorch.h"
  Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/modules"
  )
target_link_libraries(${LIB_NAME} PRIVATE ${TORCH_LIBRARIES})
target_include_directories(${LIB_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/modules>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

# Install library, create target file
install(TARGETS "${LIB_NAME}"
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

# Install target file
install(EXPORT ${PROJECT_NAME}
  FILE ${PROJECT_NAME}Config.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake
  )

# Install Fortran module files
install(FILES "${CMAKE_BINARY_DIR}/modules/ftorch.mod"
 DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}"
 )

# C++ demo
add_executable(ts_infer_cpp ts_inference.cpp)
target_link_libraries(ts_infer_cpp ${TORCH_LIBRARIES})
install(TARGETS ts_infer_cpp DESTINATION ${CMAKE_INSTALL_BINDIR})

# C demo
add_executable(ts_infer_c ts_inference.c)
target_link_libraries(ts_infer_c ${PROJECT_NAME}::${LIB_NAME})
install(TARGETS ts_infer_c DESTINATION ${CMAKE_INSTALL_BINDIR})

# Fortran demo
add_executable(ts_infer_fortran ts_inference.f90)
target_link_libraries(ts_infer_fortran PRIVATE ${PROJECT_NAME}::${LIB_NAME})
install(TARGETS ts_infer_fortran DESTINATION ${CMAKE_INSTALL_BINDIR})
