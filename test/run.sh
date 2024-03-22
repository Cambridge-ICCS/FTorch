#!/usr/bin/bash

rm -rf build
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_Fortran_COMPILER=mpif90

cmake --build .

NP=4
mpiexec -np ${NP} ./test_device_index
