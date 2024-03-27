#!/usr/bin/bash

rm -rf src/build
mkdir src/build
cd src/build

export Torch_DIR=/path/to/torch/

cmake .. -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/path/to/FTorch/src/build \
		-DENABLE_CUDA=TRUE

cmake --build .
cmake --install .
