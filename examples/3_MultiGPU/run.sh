#!/usr/bin/bash

set -eu

NP=4

rm -rf *.pt __pycache__ build simplenet.py
mkdir build

cp ../1_SimpleNet/simplenet.py .

source ../../setup.sh
python3 simplenet.py
python3 pt2ts.py
mpiexec -np ${NP} python3 simplenet_infer_python.py
deactivate

cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/path/to/FTorch/src/build
cmake --build .

mpiexec -np ${NP} ./simplenet_infer_fortran ../saved_simplenet_model_cuda.pt
