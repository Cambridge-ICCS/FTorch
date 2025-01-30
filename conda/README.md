# Conda environments for FTorch

A number of users manage software environments with conda.

Our preference in FTorch is to manage the installation of key dependencies ourselves,
with any Python dependencies managed via pip, as described in our documentation.

However, for those who wish to use conda we provide environment files and instructions
for installation here.

As it is not our preference these environments are not actively maintained.
If you experience problems please [open an issue](https://github.com/Cambridge-ICCS/FTorch/issues)
or submit a pull request if you have a fix.


## Environments

### CPU only

From a conda base environment run:
```sh
conda env create -f environment_cpu.yaml
```
from this directory to create the environment and install dependencies.

FTorch can then be built as described in the main documentation from within this
activated environment.
Note that it is convenient to install FTorch into the conda environment using
`$CONDA_PREFIX`, and locate the CMake headers for torch using the Python utility.

A CMake build command utilising this may look something like:
```sh
cmake \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUDA=FALSE \
    ..
cmake --build . --target install
```

### CUDA

From a conda base environment run:
```sh
conda env create -f environment_cuda.yaml
```
from this directory to create the environment and install dependencies.

At time of writing building FTorch in this environment requires some additional
CMake flags to be set in order to correctly locate CUDA components within the
conda environment, namely `CUDA_TOOLKIT_ROOT_DIR` and `nvtx3_dir`
(see [this comment](https://github.com/conda-forge/cuda-feedstock/issues/59#issuecomment-2620910028)
for details).
Doing this, with the tips described above for CPU builds, results in a CMake command
similar to the following:
```sh
cmake \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)') \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUDA=TRUE \
    -DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX/targets/x86_64-linux \
    -Dnvtx3_dir=$CONDA_PREFIX/targets/x86_64-linux/include/nvtx3 \
    ..
cmake --build . --target install
```

### Other Backends

We currently only provide conda environments for CPU and CUDA backends.

If you require something else please [get in touch](https://github.com/Cambridge-ICCS/FTorch/issues)
or submit a pull request.


> [!NOTE]  
> These environments will install PyTorch (and associated dependencies) to couple to
> from FTorch. Users wanting a minimal build that relies on LibTorch rather than
> PyTorch and Python should follow the non-conda build procedure.


## Examples

The instructions for running the examples make use of virtual environments
and pip to install dependencies for running any Python code within them.

This is not neccessary from within the conda environment as these dependencies are
installed as part of the environment files provided.

Conda users should adjust their approach accordingly.


## Tests

If running the unit tests it is recommended that pFUnit is build and installed into the
conda environment at `$CONDA_PREFIX`:
```sh
git clone --recursive git@github.com:Goddard-Fortran-Ecosystem/pFUnit.git

cd pFUnit
mkdir build
cd build

cmake \
  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
  ..
make tests
make install
```

The unit tests only can be run using the provided script:
```sh
./run_test_suite.sh -u
```

> [!NOTE]  
> The automated integration testing also makes use of pip to install pytorch and other
> python dependencies. Conda users wishing to run this should amend the test script
> to  remove the `pip install` command as these additional requirements are included
> in the environment files provided.
