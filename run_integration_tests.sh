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
BUILD_DIR=src/build
if [ -e "${BUILD_DIR}/test/examples/3_MultiGPU" ]
then
  EXAMPLES="1_SimpleNet 2_ResNet18 3_MultiGPU 4_MultiIO"
else
  EXAMPLES="1_SimpleNet 2_ResNet18 4_MultiIO"
fi

for EXAMPLE in ${EXAMPLES}
do
  pip -q install -r examples/${EXAMPLE}/requirements.txt
  cd ${BUILD_DIR}/test/examples/${EXAMPLE}
  ctest ${CTEST_ARGS}
  cd -
done
