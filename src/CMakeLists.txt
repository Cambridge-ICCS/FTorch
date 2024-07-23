cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
set(PROJECT_NAME FTorch)
set(LIB_NAME ftorch)
set(PACKAGE_VERSION 0.1)

project(${PROJECT_NAME} VERSION ${PACKAGE_VERSION} LANGUAGES C CXX Fortran)

include(FortranCInterface)
FortranCInterface_VERIFY(CXX QUIET)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_Fortran_FLAGS)
    set(CMAKE_Fortran_FLAGS "-std=f2008")
endif()

include(CheckLanguage)
if(ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(WARNING "No CUDA support")
  endif()
endif()

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
# Add an alias FTorch::ftorch for the library
add_library(${PROJECT_NAME}::${LIB_NAME} ALIAS ${LIB_NAME})
set_target_properties(${LIB_NAME} PROPERTIES
  PUBLIC_HEADER "ctorch.h"
  Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/modules"
  )
# Link TorchScript
target_link_libraries(${LIB_NAME} PRIVATE ${TORCH_LIBRARIES})
# Include the Fortran mod files in the library
target_include_directories(${LIB_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/modules>
#    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

# Install library, create target file
install(TARGETS "${LIB_NAME}"
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}
  )

# Install target file
install(EXPORT ${PROJECT_NAME}
  FILE ${PROJECT_NAME}Config.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
  )

# Install Fortran module files
install(FILES "${CMAKE_BINARY_DIR}/modules/ftorch.mod"
 DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}"
 )

# Build integration tests
if(CMAKE_BUILD_TESTS)
  file(MAKE_DIRECTORY test/examples)
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../examples/CMakeLists.txt
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/test/examples
    )
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../examples/1_SimpleNet
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/test/examples
    )
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../examples/2_ResNet18
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/test/examples
    )
  # file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../examples/3_MultiGPU
  #   DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/test/examples
  #   )
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../examples/4_MultiIO
    DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/test/examples
    )
  add_subdirectory(test/examples)
endif()
