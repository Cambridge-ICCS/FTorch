#!/bin/bash
# ---
# Execute this shell script to run all of FTorch's integration tests.
#
# Assumes FTorch has been built with the `-DCMAKE_BUILD_TESTS=TRUE` option.
# The `BUILD_DIR` variable in this script should be updated as appropriate for
# your configuration.
#
# See `src/test/README.md` for more details on integration testing.
# ---

set -eu

CTEST_ARGS=$@
EXAMPLES="1_SimpleNet 2_ResNet18 4_MultiIO 7_Autograd"
BUILD_DIR=src/build

for EXAMPLE in ${EXAMPLES}; do
  pip -q install -r examples/${EXAMPLE}/requirements.txt
  cd ${BUILD_DIR}/test/examples/${EXAMPLE}
  ctest ${CTEST_ARGS}
  cd -
done
