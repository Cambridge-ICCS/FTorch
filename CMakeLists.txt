cmake_minimum_required(VERSION 3.15...3.31)
set(PROJECT_NAME FTorch)
set(LIB_NAME ftorch)
set(PACKAGE_VERSION 1.0.0)

project(
  ${PROJECT_NAME}
  VERSION ${PACKAGE_VERSION}
  DESCRIPTION "A library for directly calling PyTorch ML models from Fortran"
  LANGUAGES C CXX Fortran)

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

if(WIN32)
  message(
    WARNING "Windows builds are untested, so functionality is not guaranteed")
  # if building on windows we need to make sure the symbols are exported
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
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
set(GPU_DEVICE_HIP  1)  # NOTE: HIP is treated as CUDA in FTorch
set(GPU_DEVICE_XPU 12)
set(GPU_DEVICE_MPS 13)
option(GPU_DEVICE
       "Set the GPU device (NONE [default], CUDA, HIP, XPU, or MPS)" NONE)
if("${GPU_DEVICE}" STREQUAL "OFF")
  set(GPU_DEVICE NONE)
endif()

if("${GPU_DEVICE}" STREQUAL "NONE")
  message(STATUS "CPU-only build")
  set(GPU_DEVICE_CODE ${GPU_DEVICE_NONE})
elseif("${GPU_DEVICE}" STREQUAL "CUDA")
  message(STATUS "Building with CUDA support")
  set(GPU_DEVICE_CODE ${GPU_DEVICE_CUDA})
elseif("${GPU_DEVICE}" STREQUAL "HIP")
  # As stated in the PyTorch documentation
  # (https://docs.pytorch.org/docs/stable/notes/hip.html) > "PyTorch for HIP
  # intentionally reuses the existing torch.cuda interfaces.
  # > This helps to accelerate the porting of existing PyTorch code and models
  # > because very few code changes are necessary, if any."
  # Therefore we use the CUDA backend when HIP is selected. This is technically
  # equivalent to specifying -DGPU_DEVICE=CUDA.
  message(STATUS "Building with HIP support")
  set(GPU_DEVICE_CODE ${GPU_DEVICE_HIP})
elseif("${GPU_DEVICE}" STREQUAL "XPU")
  message(STATUS "Building with XPU support")
  set(GPU_DEVICE_CODE ${GPU_DEVICE_XPU})
elseif("${GPU_DEVICE}" STREQUAL "MPS")
  message(STATUS "Building with MPS support")
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
if("${GPU_DEVICE}" STREQUAL "HIP")
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
  else()
    message(WARNING "No HIP support")
  endif()
endif()

# Set RPATH behaviour
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_BUILD_RPATH "${INSTALL_PREFIX}")
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
# Embed absolute paths to external libraries that are not part of the project,
# (they are expected to be at the same location on all machines the project will
# be deployed to
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Follow GNU conventions for installing directories
include(GNUInstallDirs)

# Dependencies
find_package(Torch REQUIRED)

# Define RPATH for executables via a relative expression to enable a fully
# relocatable package
file(RELATIVE_PATH relDir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
     ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
get_filename_component(TORCH_LIBRARY_DIR ${TORCH_LIBRARY} DIRECTORY)
set(CMAKE_INSTALL_RPATH "$ORIGIN/${relDir};${TORCH_LIBRARY_DIR}")

# Initialize Fortran module build output directory for all subsequent targets
if(NOT DEFINED CMAKE_Fortran_MODULE_DIRECTORY)
  set(CMAKE_Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/modules")
endif()

# Library with C and Fortran bindings
add_library(${LIB_NAME} src/ctorch.cpp src/ftorch.F90 src/ftorch_test_utils.f90)

# Define compile definitions, including GPU devices
target_compile_definitions(
  ${LIB_NAME}
  PRIVATE GPU_DEVICE=${GPU_DEVICE_CODE}
          GPU_DEVICE_NONE=${GPU_DEVICE_NONE}
          GPU_DEVICE_CUDA=${GPU_DEVICE_CUDA}
          GPU_DEVICE_HIP=${GPU_DEVICE_HIP}
          GPU_DEVICE_XPU=${GPU_DEVICE_XPU}
          GPU_DEVICE_MPS=${GPU_DEVICE_MPS})

# Add an alias FTorch::ftorch for the library
add_library(${PROJECT_NAME}::${LIB_NAME} ALIAS ${LIB_NAME})
# cmake-format: off
set_target_properties(${LIB_NAME} PROPERTIES PUBLIC_HEADER "src/ctorch.h")
# cmake-format: on
# Link TorchScript
# NOTE: Linking stdc++ for downstream targets fixes ctorch link error
target_link_libraries(${LIB_NAME}
  PRIVATE ${TORCH_LIBRARIES} INTERFACE stdc++)
# Include the Fortran mod files in the library
target_include_directories(
  ${LIB_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_Fortran_MODULE_DIRECTORY}>
  # $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Install library, create target file
install(
  TARGETS "${LIB_NAME}"
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PRIVATE_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME})

# Make targets available in build dir
export(
  EXPORT ${PROJECT_NAME}Targets
  NAMESPACE ${PROJECT_NAME}::
  FILE ${CMAKE_BINARY_DIR}/${PROJECT_NAME}Targets.cmake)

# Install target file
install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

# Install Fortran module files
if(NOT DEFINED CMAKE_INSTALL_MODULEDIR)
  set(
    CMAKE_INSTALL_MODULEDIR
    "${CMAKE_INSTALL_INCLUDEDIR}/${LIB_NAME}"
    CACHE STRING "Directory in prefix to install generated module files"
  )
endif()

install(FILES "${CMAKE_Fortran_MODULE_DIRECTORY}/ftorch.mod"
        DESTINATION "${CMAKE_INSTALL_MODULEDIR}")
install(FILES "${CMAKE_Fortran_MODULE_DIRECTORY}/ftorch_test_utils.mod"
        DESTINATION "${CMAKE_INSTALL_MODULEDIR}")

# Build integration tests
if(CMAKE_BUILD_TESTS)

  set(Python_FIND_VIRTUALENV FIRST) # cmake-lint: disable=C0103
  find_package(
    Python
    COMPONENTS Interpreter
    REQUIRED)

  # Check if Python is in a virtual environment by checking the VIRTUAL_ENV
  # environment variable exists
  if(NOT DEFINED ENV{VIRTUAL_ENV} AND NOT DEFINED ENV{CONDA_PREFIX})
    message(FATAL_ERROR
            "No Python virtual environment detected. Please activate one.")
  endif()

  if(NOT "${GPU_DEVICE}" STREQUAL "NONE")
    # If a GPU device is passed, specify if more than one is available. Set to
    # OFF if only one is available. Used to determine whether to run Multi GPU
    # testing
    option(MULTI_GPU "Whether to enable Multi GPU testing (ON [default], OFF)"
           ON)
  endif()

  # Enable CTest
  enable_testing()

  # Unit tests
  # NOTE: We do not currently support unit testing on Windows
  if(UNIX)
    add_subdirectory(test/unit)
  endif()

  # Integration tests
  add_subdirectory(examples)
endif()

# Setup and install configuration files
# NOTE: Must occur after BUILD_TESTS because find_package(FTorch QUIET) breaks
# otherwise
get_filename_component(TORCH_BASE_DIR "${TORCH_LIBRARY_DIR}" DIRECTORY)
include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
  PATH_VARS TORCH_BASE_DIR)

install(
  FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

# Export a pkg-config file
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/template.pc"
  "${PROJECT_BINARY_DIR}/${LIB_NAME}.pc"
  @ONLY
)
install(
  FILES
  "${PROJECT_BINARY_DIR}/${LIB_NAME}.pc"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
)
