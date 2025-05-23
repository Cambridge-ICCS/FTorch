#!/bin/bash
# ---
# Script to build and install FTorch into a codespace environment
# It should be run from the top of the repository i.e. `FTorch/`
# ---

# Set up a virtual environment an install neccessary Python dependencies
#   We will specify the cpu-only version of PyTorch to match the codespace hardware
python3 -m venv venv
# shellcheck source=/dev/null
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Extract the location of the installed packages in the venv
# This is typically `FTorch/build/venv/lib/python3.xx/site-packages/`
PYTHON_PATH=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")

# Create a build directory to build FTorch in using CMake
mkdir build
cd build || exit

# Build FTorch using CMake
#   We will build in `Release`, linking to the libtorch installed in the virtual
#   environment, and installing the final library into FTorch/bin/
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PYTHON_PATH"/torch \
  -DCMAKE_INSTALL_PREFIX=/workspaces/FTorch/bin
cmake --build . --target install

# Add LibTorch libraries to the paths to be searched for dynamic linking at runtime
export LD_LIBRARY_PATH=$PYTHON_PATH/torch/lib

# return user to the root of the project
cd ../
