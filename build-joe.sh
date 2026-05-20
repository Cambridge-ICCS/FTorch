#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status

# Function to display help text
show_help() {
  echo "Usage: $0 [--debug | -d] [--fresh | -f] [--fpm] [--std <standard>] [--help | -h]"
  echo
  echo "Options:"
  echo "  --debug   | -d      Compile in Debug mode."
  echo "  --fresh   | -f      Create a fresh build before compiling."
  echo "  --fpm     | -fpm    Use Fortran Package Manager (fpm) for building"
  echo "                      instead of CMake."
  echo "  --std <standard>    Specify the Fortran standard (e.g., f2008, f2018)."
  echo "  --help    | -h      Show this help message and exit."
}

# Check if a Python virtual environment is active
if [ -z "${VIRTUAL_ENV}" ]; then
  echo "No virtual environment is active. Please activate a virtual environment \
before running this script."
  exit 1
fi

# Install ftorch_utils
pip install -q -e .[examples] --extra-index-url https://download.pytorch.org/whl/cpu

# # Check if a Spack environment is active
# if [ -z "${SPACK_ENV}" ]; then
#   echo "No Spack environment is active. Please activate a Spack environment \
# before running this script."
#   exit 1
# fi

# Parse command line arguments
BUILD_DIR="$(pwd)/build"
BUILD_TYPE=Release
FFLAGS=""
FRESH_BUILD=false
FPM_BUILD=false
FORTRAN_STANDARD=f2018
HELP=false
for arg in "$@"; do
  case $arg in
  --debug | -d)
    BUILD_DIR="${BUILD_DIR}_debug"
    BUILD_TYPE=Debug
    shift
    ;;
  --fresh | -f)
    FRESH_BUILD=true
    shift
    ;;
  --fpm | -fpm)
    FPM_BUILD=true
    shift
    ;;
  --std)
    FORTRAN_STANDARD="$2"
    shift 2
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
  FFLAGS="${FFLAGS} -std=${FORTRAN_STANDARD}"
elif [ -n "$(mpif90 --version | grep IFX)" ]; then
  BUILD_DIR="${BUILD_DIR}/intel"
  FFLAGS="${FFLAGS} -stand ${FORTRAN_STANDARD} -fpscomp logicals"
elif [ -n "$(mpif90 --version | grep NVIDIA)" ]; then
  BUILD_DIR="${BUILD_DIR}/nvidia"
  # FFLAGS="${FFLAGS} -std=${FORTRAN_STANDARD}"
elif [ -n "$(mpif90 --version | grep flang)" ]; then
  BUILD_DIR="${BUILD_DIR}/flang"
  FFLAGS="${FFLAGS} -std=${FORTRAN_STANDARD}"
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

# Check if a fresh build is requested
if [ "${FRESH_BUILD}" = true ]; then
  echo "Creating a fresh build..."
  if [ "${FPM_BUILD}" = true ]; then
    fpm clean
  fi
  rm -rf "${BUILD_DIR}"
else
  echo "Rebuilding..."
fi
mkdir -p "${BUILD_DIR}"

# Paths to dependencies
Torch_DIR="${VIRTUAL_ENV}/lib/python3.14/site-packages"
export SPACK_COLOR=false
PFUNIT_DIR="$(spack find --format 'pfunit-{version}-{hash}/PFUNIT-{version}' pfunit)"
export SPACK_COLOR=always
PFUNIT_DIR="${SPACK_ROOT}/opt/spack/linux-skylake/${PFUNIT_DIR::-2}"

# Build the library
if [ "${FPM_BUILD}" = true ]; then
  fpm build \
    --profile debug \
    --compiler mpif90 \
    --c-compiler mpicc \
    --cxx-compiler mpicxx \
    --flag "${FFLAGS}" \
    --c-flag "-cpp -std=c++17 -I${Torch_DIR}/include -I${Torch_DIR}/include/torch/csrc/api/include" \
    --cxx-flag "-cpp -std=c++17 -I${Torch_DIR}/include -I${Torch_DIR}/include/torch/csrc/api/include" \
    --link-flag "-L${Torch_DIR}/lib -L${PFUNIT_DIR}/lib" \
    --verbose
else
  cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_Fortran_COMPILER=mpif90 \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}" \
    -DCMAKE_BUILD_TESTS=TRUE \
    -DCMAKE_PREFIX_PATH="${PFUNIT_DIR};${Torch_DIR}" \
    -DCMAKE_Fortran_FLAGS="${FFLAGS}"
  cmake --build "${BUILD_DIR}" --config Release --verbose
  cmake --install "${BUILD_DIR}"
fi
