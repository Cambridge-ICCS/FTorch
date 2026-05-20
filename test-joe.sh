#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status

# Function to display help text
show_help() {
  echo "Usage: $0 [--debug | -d] [--fresh | -f] [--fpm] [--std <standard>]"
  echo "                  [--compiler <compiler>] [--help | -h]"
  echo
  echo "Options:"
  echo "  --debug   | -d      Run in Debug mode."
  echo "  --fpm     | -fpm    Use Fortran Package Manager (fpm) build"
  echo "                      instead of CMake."
  echo "  --help    | -h      Show this help message and exit."
}

# Check if a Python virtual environment is active
if [ -z "${VIRTUAL_ENV}" ]; then
  echo "No virtual environment is active. Please activate a virtual environment \
before running this script."
  exit 1
fi

# Parse command line arguments
BUILD_DIR="$(pwd)/build"
BUILD_TYPE=Release
FPM_BUILD=false
HELP=false
for arg in "$@"; do
  case $arg in
  --debug | -d)
    BUILD_DIR="${BUILD_DIR}_debug"
    BUILD_TYPE=Debug
    shift
    ;;
  --fpm | -fpm)
    FPM_BUILD=true
    shift
    ;;
  --help | -h)
    HELP=true
    shift
    ;;
  *) ;;
  esac
done

# Determine compiler family
if [ -n "$(mpif90 --version | grep GNU)" ]; then
  BUILD_DIR="${BUILD_DIR}/gnu"
elif [ -n "$(mpif90 --version | grep IFX)" ]; then
  BUILD_DIR="${BUILD_DIR}/intel"
elif [ -n "$(mpif90 --version | grep NVIDIA)" ]; then
  BUILD_DIR="${BUILD_DIR}/nvidia"
elif [ -n "$(mpif90 --version | grep flang)" ]; then
  BUILD_DIR="${BUILD_DIR}/flang"
else
  echo "Unsupported compiler: ${COMPILER}"
  echo "Supported compilers are: gnu, intel, nvidia, flang"
  exit 1
fi

# Check for --help option
if [ "${HELP}" = true ]; then
  show_help
  exit 0
fi

# Run Fortran tests
if [ "${FPM_BUILD}" = true ]; then
  fpm test
else
  cd "${BUILD_DIR}" || exit
  ctest -V "$@"
  cd - || exit
fi

# Run Python tests
if [ "$@" != "" ]; then
  pytest -v test/ftorch_utils
fi
