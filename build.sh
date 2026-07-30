#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status

# Function to display help text
show_help() {
  echo "Usage: $0 [--debug | -d] [--fresh | -f] [--fpm] [--std <standard>] [--help | -h]"
  echo
  echo "Options:"
  echo "  --test    | -t      Build and run tests."
  echo "  --clean   | -c      Create a fresh build before compiling."
  echo "  --debug   | -d      Compile in Debug mode."
  echo "  --help    | -h      Show this help message and exit."
}


# Set Compilers
FC=$(brew --prefix)/bin/gfortran
FFLAGS=""
CC=clang
CFLAGS=""
CXX=clang++
CXXFLAGS=""
LFLAGS=""
# Let CMake's FindOpenMP locate Homebrew's libomp for pFUnit, without
# polluting global link flags (which would clash with PyTorch's bundled libomp).
OPENMP_ROOT="$(brew --prefix libomp)"


# Parse command line arguments
ROOT_DIR="$(pwd)"
BUILD_DIR="$(pwd)/build"
BUILD_TYPE=Release
TEST_BUILD=false
PFUNIT_DIR=""
INSTALL_DIR="$(pwd)/ftorch_install"
CLEAN_BUILD=false
CLEAN_PFUNIT=false
FORTRAN_STANDARD=f2008
HELP=false
for arg in "$@"; do
  case $arg in
  --test | -t)
    TEST_BUILD=true
    shift
    ;;
  --clean | -c)
    CLEAN_BUILD=true
    shift
    ;;
  --clean-pfunit)
    CLEAN_PFUNIT=true
    shift
    ;;
  --debug | -d)
    BUILD_DIR="${BUILD_DIR}_debug"
    BUILD_TYPE=Debug
    shift
    ;;
  --std)
    FORTRAN_STANDARD="$2"
    shift 2
    FFLAGS="${FFLAGS} -std=${FORTRAN_STANDARD}"
    ;;
  --help | -h)
    HELP=true
    shift
    ;;
  *) ;;
  esac
done


# Set up directories
if [ "${CLEAN_BUILD}" = true ]; then
  echo "Creating a clean build..."
  rm -rf "${BUILD_DIR}"
else
  echo "Rebuilding..."
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

if [ "${CLEAN_BUILD}" = true ]; then
  # python3 -m venv ftorch_venv
  # source ftorch_venv/bin/activate
  # pip install --editable ../.[examples]
  uv venv ftorch_venv
  source ftorch_venv/bin/activate
  uv pip install --editable ../.[examples]
else
  source ftorch_venv/bin/activate
fi

if [ -z "${VIRTUAL_ENV}" ]; then
  echo "There is an issue activating the virtual environment."
  exit 1
fi


# pFUnit
cd "${ROOT_DIR}"
if [ "${TEST_BUILD}" = true ]; then
  if [[ ! -d "pFUnit" || "${CLEAN_PFUNIT}" = true ]]; then
    echo "Building pFUnit..."
    rm -rf pFUnit/
    git clone -b v4.12.0 https://github.com/Goddard-Fortran-Ecosystem/pFUnit.git
    mkdir pFUnit/build
    cd pFUnit
    cmake -S . -B build \
      -DCMAKE_Fortran_COMPILER="${FC}"
    cmake --build build --target tests -j 4
    ctest --test-dir build --output-on-failure
    cmake --build build --target install -j 4
    cd ../
  fi
  # NOTE: The pFUnit version (pinned during installation above) is used in the install path.
  PFUNIT_DIR=$(pwd)/pFUnit/build/installed/PFUNIT-4.12
fi


# FTorch
cd "${ROOT_DIR}"
VN=$(python -c "import sys; print('.'.join(sys.version.split('.')[:2]))")
Torch_DIR="${VIRTUAL_ENV}/lib/python${VN}/site-packages"

cmake -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DCMAKE_C_COMPILER=$CC  \
  -DCMAKE_CXX_COMPILER=$CXX  \
  -DCMAKE_Fortran_COMPILER=$FC \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DCMAKE_BUILD_TESTS="${TEST_BUILD}" \
  -DFTORCH_BUILD_UNIT_TESTS="${TEST_BUILD}" \
  -DFTORCH_BUILD_INTEGRATION_TESTS="${TEST_BUILD}" \
  -DCMAKE_PREFIX_PATH="${PFUNIT_DIR};${Torch_DIR}" \
  -DOpenMP_ROOT="${OPENMP_ROOT}" \
  -DCMAKE_Fortran_FLAGS="${FFLAGS}" \
  -DCMAKE_C_FLAGS="${CFLAGS}" \
  -DCMAKE_CXX_FLAGS="${CXXFLAGS}" \
  -DCMAKE_EXE_LINKER_FLAGS="${LFLAGS}" \
  -DGPU_DEVICE=MPS
cmake --build "${BUILD_DIR}" -j 4
cmake --install "${BUILD_DIR}"

# Tests
if [ "${TEST_BUILD}" = true ]; then
  ctest --test-dir "${BUILD_DIR}" --verbose --tests-regex unit
  ctest --test-dir "${BUILD_DIR}" --verbose --tests-regex example
  ctest --test-dir "${BUILD_DIR}" --verbose --tests-regex testapp_tensor_constructors
fi
