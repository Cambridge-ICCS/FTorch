#!/bin/bash

#cd fortran-pytorch-lib
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=${BUILD_PREFIX}/lib/python${PY_VER}/site-packages/torch/share/cmake -DCMAKE_INSTALL_PREFIX=$PREFIX ..

make
make install
