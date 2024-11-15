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

BUILD_DIR=src/build

# Unit tests
cd ${BUILD_DIR}/test/unit
ctest ${CTEST_ARGS}
cd -

# Integration tests
EXAMPLES="1_SimpleNet 2_ResNet18 4_MultiIO 6_Autograd"
for EXAMPLE in ${EXAMPLES}; do
  pip -q install -r examples/"${EXAMPLE}"/requirements.txt
  cd "${BUILD_DIR}"/test/examples/"${EXAMPLE}"
  ctest "$@"
  cd -
done
