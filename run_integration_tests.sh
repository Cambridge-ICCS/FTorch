#!/usr/bin/bash
# ---
# Execute this shell script to run all of FTorch's integration tests.
#
# Assumes FTorch has been built with the `-DCMAKE_BUILD_TESTS=TRUE` option.
#
# See `src/test/README.md` for more details on integration testing.
# ---

set -eux

CTEST_ARGS=$@
EXAMPLES="1_SimpleNet 2_ResNet18"

cd src/build/test/examples
for EXAMPLE in ${EXAMPLES}
do
  cd ${EXAMPLE}
  ctest ${CTEST_ARGS}
  cd -
done
