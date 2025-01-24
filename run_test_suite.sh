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
  echo "Usage: $0 [BUILD_DIR=<build_dir>] [--integration-only | i] [--unit-only | -u]"
  echo "          [--verbose | -V] [--help | h]"
  echo
  echo "Options:"
  echo "  BUILD_DIR=<build_dir> Specify the build directory (default: src/build)."
  echo "  --integration-only | -i        Run integration tests only."
  echo "  --unit-only        | -u        Run unit tests only."
  echo "  --verbose          | -V        Run with verbose ctest output."
  echo "  --help             | -h        Show this help message and exit."
}

# Parse command line arguments
BUILD_DIR="src/build"
INTEGRATION_ONLY=false
UNIT_ONLY=false
VERBOSE=false
HELP=false
for ARG in "$@"; do
  case ${ARG} in
  BUILD_DIR=*)
    BUILD_DIR="${ARG#*=}"
    ;;
  --integration-only | -i)
    INTEGRATION_ONLY=true
    shift
    ;;
  --unit-only | -u)
    UNIT_ONLY=true
    shift
    ;;
  --verbose | -V)
    VERBOSE=true
    shift
    ;;
  --help | -h)
    HELP=true
    shift
    ;;
  *)
    echo "Unknown argument: ${ARG}"
    show_help
    exit 1
    ;;
  esac
done

# Check for --help option
if [ "${HELP}" = true ]; then
  show_help
  exit 0
fi

# Process command line arguments
if [ "${VERBOSE}" = true ]; then
  CTEST_ARGS="-V"
else
  CTEST_ARGS=""
fi

# Run unit tests
if [ "${INTEGRATION_ONLY}" = false ]; then
  cd "${BUILD_DIR}/test/unit"
  ctest "${CTEST_ARGS}"
  cd -
fi

# Run integration tests
if [ "${UNIT_ONLY}" = false ]; then
  if [ -e "${BUILD_DIR}/test/examples/3_MultiGPU" ]; then
    EXAMPLES="1_SimpleNet 2_ResNet18 3_MultiGPU 4_MultiIO 6_Autograd"
  else
    EXAMPLES="1_SimpleNet 2_ResNet18 4_MultiIO 6_Autograd"
  fi
  for EXAMPLE in ${EXAMPLES}; do
    pip -q install -r examples/"${EXAMPLE}"/requirements.txt
    cd "${BUILD_DIR}"/test/examples/"${EXAMPLE}"
    ctest "${CTEST_ARGS}"
    cd -
  done
fi
