cmake_minimum_required(VERSION 3.15...3.31)
set(PROJECT_NAME FTorch)
set(LIB_NAME ftorch)
set(PACKAGE_VERSION 1.0.0)

project(
  ${PROJECT_NAME}
  VERSION ${PACKAGE_VERSION}
  LANGUAGES C CXX Fortran)

if(WIN32)
  # if building on windows we need to make sure the symbols are exported
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  # required to build the .dll on windows
  set(BUILD_SHARED_LIBS TRUE)
endif()

include(FortranCInterface)
FortranCInterface_VERIFY(CXX QUIET)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set GPU device type using consistent numbering as in PyTorch
# https://github.com/pytorch/pytorch/blob/main/c10/core/DeviceType.h
# Set in a single location here and passed as preprocessor flag for use
# throughout source files.
set(GPU_DEVICE_NONE 0)
set(GPU_DEVICE_CUDA 1)
set(GPU_DEVICE_XPU 12)
set(GPU_DEVICE_MPS 13)
option(GPU_DEVICE "Set the GPU device (NONE [default], CUDA, XPU, or MPS)" NONE)
if("${GPU_DEVICE}" STREQUAL "OFF")
  set(GPU_DEVICE NONE)
endif()
if("${GPU_DEVICE}" STREQUAL "NONE")
  set(GPU_DEVICE_CODE ${GPU_DEVICE_NONE})
elseif("${GPU_DEVICE}" STREQUAL "CUDA")
  set(GPU_DEVICE_CODE ${GPU_DEVICE_CUDA})
elseif("${GPU_DEVICE}" STREQUAL "XPU")
  set(GPU_DEVICE_CODE ${GPU_DEVICE_XPU})
elseif("${GPU_DEVICE}" STREQUAL "MPS")
  set(GPU_DEVICE_CODE ${GPU_DEVICE_MPS})
else()
  message(SEND_ERROR "GPU_DEVICE '${GPU_DEVICE}' not recognised")
endif()

# Other GPU specific setup
include(CheckLanguage)
if("${GPU_DEVICE}" STREQUAL "CUDA")
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
# Embed absolute paths to external libraries that are not part of the project,
# (they are expected to be at the same location on all machines the project will
# be deployed to
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Follow GNU conventions for installing directories
include(GNUInstallDirs)

# Define RPATH for executables via a relative expression to enable a fully
# relocatable package
file(RELATIVE_PATH relDir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
     ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_RPATH $ORIGIN/${relDir})

# Dependencies
find_package(Torch REQUIRED)

# Library with C and Fortran bindings
add_library(${LIB_NAME} SHARED src/ctorch.cpp src/ftorch.F90
                               src/ftorch_test_utils.f90)

# Define compile definitions, including GPU devices
set(COMPILE_DEFS "")
if(UNIX AND NOT APPLE)
  # only add UNIX definition for linux (not apple which is also unix)
  set(COMPILE_DEFS UNIX)
endif()
target_compile_definitions(
  ${LIB_NAME}
  PRIVATE ${COMPILE_DEFS} GPU_DEVICE=${GPU_DEVICE_CODE}
          GPU_DEVICE_NONE=${GPU_DEVICE_NONE} GPU_DEVICE_CUDA=${GPU_DEVICE_CUDA}
          GPU_DEVICE_XPU=${GPU_DEVICE_XPU} GPU_DEVICE_MPS=${GPU_DEVICE_MPS})

# Add an alias FTorch::ftorch for the library
add_library(${PROJECT_NAME}::${LIB_NAME} ALIAS ${LIB_NAME})
# cmake-format: off
set_target_properties(
  ${LIB_NAME} PROPERTIES PUBLIC_HEADER "src/ctorch.h"
  Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/modules")
# cmake-format: on
# Link TorchScript
target_link_libraries(${LIB_NAME} PRIVATE ${TORCH_LIBRARIES})
# Include the Fortran mod files in the library
target_include_directories(
  ${LIB_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/modules>
  # $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Install library, create target file
install(
  TARGETS "${LIB_NAME}"
  EXPORT ${PROJECT_NAME}
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME})

# Install target file
install(
  EXPORT ${PROJECT_NAME}
  FILE ${PROJECT_NAME}Config.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

# Install Fortran module files
install(FILES "${CMAKE_BINARY_DIR}/modules/ftorch.mod"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}")
install(FILES "${CMAKE_BINARY_DIR}/modules/ftorch_test_utils.mod"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}")

# Build integration tests
if(CMAKE_BUILD_TESTS)

  set(Python_FIND_VIRTUALENV FIRST) # cmake-lint: disable=C0103
  find_package(
    Python
    COMPONENTS Interpreter
    REQUIRED)

  enable_testing()

  # Unit tests
  # NOTE: We do not currently support unit testing on Windows
  if(UNIX)
    add_subdirectory(test/unit)
  endif()

  # Integration tests
  add_subdirectory(examples)
endif()
