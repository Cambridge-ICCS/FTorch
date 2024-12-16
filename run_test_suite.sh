#!/bin/bash
# ---
# Execute this shell script to run FTorch's test suite. This includes both unit
# tests and integration tests.
#
# Assumes FTorch has been built with the `-DCMAKE_BUILD_TESTS=TRUE` option.
# The `BUILD_DIR` variable in this script should be updated as appropriate for
# your configuration.
#
# See `src/test/README.md` for more details on the test suite.
# ---

set -eu

# Function to display help text
show_help() {
  echo "Usage: $0 [BUILD_DIR=<build_dir>] [--verbose | -V]"
  echo
  echo "Options:"
  echo "  BUILD_DIR=<build_dir> Specify the build directory (default: src/build)."
  echo "  --verbose | -V        Run with verbose ctest output."
  echo "  --help                Show this help message and exit."
}

# Check for --help option
if [ "$1" = "--help" ]; then
  show_help
  exit 0
fi

# Parse command line arguments
BUILD_DIR="src/build"
VERBOSE=false
for ARG in "$@"; do
  case ${ARG} in
  BUILD_DIR=*)
    BUILD_DIR="${ARG#*=}"
    ;;
  --verbose | -V)
    VERBOSE=true
    shift
    ;;
  *)
    echo "Unknown argument: ${ARG}"
    show_help
    exit 1
    ;;
  esac
done

# Process command line arguments
if [ "${VERBOSE}" = true ]; then
  CTEST_ARGS="-V"
else
  CTEST_ARGS=""
fi

# Run unit tests
cd ${BUILD_DIR}/test/unit
ctest ${CTEST_ARGS}
cd -

# Run integration tests
EXAMPLES="1_SimpleNet 2_ResNet18 4_MultiIO 6_Autograd"
for EXAMPLE in ${EXAMPLES}; do
  pip -q install -r examples/"${EXAMPLE}"/requirements.txt
  cd "${BUILD_DIR}"/test/examples/"${EXAMPLE}"
  ctest "$@"
  cd -
done
