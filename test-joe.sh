#!/bin/bash

set -e

cd build || exit
ctest -V "$@"
cd - || exit
if [ "$@" != "" ]; then
  pytest -v test/ftorch_utils
fi
