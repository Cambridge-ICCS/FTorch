#!/usr/bin/bash

source setup.sh

rm -rf src/build
mkdir src/build
cd src/build

cmake .. -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/path/to/FTorch/src/build \
        -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_Fortran_COMPILER=mpif90

cmake --build .
cmake --install .
