# FTorch

**_A library for coupling (Py)Torch machine learning models to Fortran_**

![GitHub](https://img.shields.io/github/license/Cambridge-ICCS/FTorch)
![Fortran](https://img.shields.io/badge/Fortran-2008-purple)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.07602/status.svg)](https://doi.org/10.21105/joss.07602)

This repository contains code, utilities, and examples for directly calling PyTorch ML
models from Fortran.

For full API and user documentation please see the
[online documentation](https://cambridge-iccs.github.io/FTorch/) which is 
significantly more detailed than this README.

To cite use of this code please refer to
[Atkinson et al., (2025) DOI:10.21105/joss.07602](https://doi.org/10.21105/joss.07602). See
[acknowledgment](#authors-and-acknowledgment) below for more details.


## Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [GPU Support](#gpu-support)
- [Examples](#examples)
- [License](#license)
- [Contributions](#contributions)
- [Authors and Acknowledgment](#authors-and-acknowledgment)
- [Users](#used-by)

## Description

It is desirable to be able to run machine learning (ML) models directly in Fortran.
Such models are often trained in some other language (say Python) using popular
frameworks (say PyTorch) and saved.
We want to run inference on this model without having to call a Python executable.
To achieve this we use the existing Torch C++ interface.

This project provides a library enabling a user to directly couple their PyTorch
models to Fortran code.
We provide installation instructions for the library as well as instructions and
examples for performing coupling.

```fortran
use ftorch
...
type(torch_model) :: model
type(torch_tensor), dimension(n_inputs)  :: model_inputs_arr
type(torch_tensor), dimension(n_outputs) :: model_output_arr
...
call torch_model_load(model, "/my/saved/TorchScript/model.pt", torch_kCPU)
call torch_tensor_from_array(model_inputs_arr(1), input_fortran, in_layout, torch_kCPU)
call torch_tensor_from_array(model_output_arr(1), output_fortran, out_layout, torch_kCPU)

call torch_model_forward(model, model_input_arr, model_output_arr)
```

The following presentations provide an introduction and overview of _FTorch_:

* FTorch: Facilitating Hybrid Modelling\
  N8-CIR Seminar, Leeds - July 2025\
  [Slides](https://jackatkinson.net/slides/Leeds-N8-FTorch) - [Recording](https://www.youtube.com/watch?v=je7St0t_W9A)
* Coupling Machine Learning to Numerical (Climate) Models\
  Platform for Advanced Scientific Computing, Zurich - June 2024\
  [Slides](https://jackatkinson.net/slides/PASC24)

If you are interested in using this library please get in touch.

_For a similar approach to calling TensorFlow models from Fortran please see [Fortran-TF-lib](https://github.com/Cambridge-ICCS/fortran-tf-lib)._

## Installation

### Dependencies

* CMake >= 3.15
* Fortran 2008, C++17, and C compilers
* [LibTorch](https://pytorch.org/cppdocs/installing.html) or [PyTorch](https://pytorch.org/)

### Build and Install

Detailed installation instructions are provided in the
[online installation documentation](https://cambridge-iccs.github.io/FTorch/page/installation/index.html).

The following instructions assume a Unix system.
For installation on Windows, Apple Silicon, Conda Environments, or Codespaces refer to
the [system-specific guidance](https://cambridge-iccs.github.io/FTorch/page/installation/systems.html).

The installation process has three main steps:

1. Fetch the source code from GitHub via git,
2. Navigate to the root FTorch directory and create a build directory,
3. Build and install the library using CMake with the relevant options.

```bash
git clone https://github.com/Cambridge-ICCS/FTorch.git
cd FTorch/
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=</path/to/torch/installation>
cmake --build . --target install
```
This will place the library files at the install location which can be tailored using
the `CMAKE_INSTALL_PREFIX` option.


The following table of CMake options are available to be passed as arguments to `cmake` through `-D<Option>=<Value>`.  
It is likely that you will need to provide at least `CMAKE_PREFIX_PATH`.

| Option                                                                                            | Value                                   | Description                                                                                                                                                                |
| ------------------------------------------------------------------------------------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`CMAKE_Fortran_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html) | `gfortran` / `ifx` / `ifort`            | Specify a Fortran compiler to build the library with. This should match the Fortran compiler you're using to build the code you are calling this library from.             |
| [`CMAKE_C_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html)       | `gcc` / `icx` / `icc`                   | Specify a C compiler to build the library with.                                                                                                                            |
| [`CMAKE_CXX_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html)     | `g++` / `icx` / `icpc`                  | Specify a C++ compiler to build the library with.                                                                                                                          |
| [`CMAKE_PREFIX_PATH`](https://cmake.org/cmake/help/latest/variable/CMAKE_PREFIX_PATH.html)        | `</path/to/LibTorch/>`                  | Location of Torch installation<sup>1</sup>                                                                                                                                 |
| [`CMAKE_INSTALL_PREFIX`](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html)  | `</path/to/install/lib/at/>`            | Location at which the library files should be installed. By default this is `/usr/local`                                                                                   |
| [`CMAKE_BUILD_TYPE`](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)          | `Release` / `Debug`                     | Specifies build type. The default is `Debug`, use `Release` for production code                                                                                            |
| [`BUILD_SHARED_LIBS`](https://cmake.org/cmake/help/latest/variable/BUILD_SHARED_LIBS.html)        | `ON` / `OFF`                            | Specifies whether to build FTorch as a shared library. The default is `ON`, use `OFF` to build FTorch as a static library.                                                 |
| `CMAKE_BUILD_TESTS`                                                                               | `TRUE` / `FALSE`                        | Specifies whether to compile FTorch's [test suite](https://cambridge-iccs.github.io/FTorch/page/testing.html) as part of the build.                                        |
| `GPU_DEVICE`                                                                                      | `NONE` / `CUDA` / `HIP` / `XPU` / `MPS` | Specifies the target GPU backend architecture (if any)                                                                                                                     |
| `MULTI_GPU`                                                                                       | `ON` / `OFF`                            | Specifies whether to build the tests that involve multiple GPU devices (`ON` by default if `CMAKE_BUILD_TESTS` and `GPU_DEVICE` are set).                                  |

<sup>1</sup> _The path to the Torch installation needs to allow CMake to locate the relevant Torch CMake files.  
      If Torch has been [installed as LibTorch](https://pytorch.org/cppdocs/installing.html)
      then this should be the absolute path to the unzipped LibTorch distribution.
      If Torch has been installed as PyTorch in a Python [venv (virtual environment)](https://docs.python.org/3/library/venv.html),
      e.g. with `pip install torch`, then this should be `</path/to/venv/>lib/python<3.xx>/site-packages/torch/`.  


## Usage

In order to use FTorch users will typically need to follow these steps:

1. Save a PyTorch model as [TorchScript](https://pytorch.org/docs/stable/jit.html).
2. Write Fortran using the FTorch bindings to use the model from within Fortran.
3. Build and compile the code, linking against the FTorch library

These steps are described in more detail in the
[online documentation](https://cambridge-iccs.github.io/FTorch/page/examples.html#overview-of-the-interfacing-process)


## GPU Support

FTorch supports running on multiple GPU hardwares including CUDA (NVIDIA),
HIP (AMD/ROCm), MPS (Apple Silicon) and XPU (Intel). It also supports running on
multiple devices.

For detailed guidance about running on GPU, including instructions for using multiple
devices, please see the
[online GPU documentation](https://cambridge-iccs.github.io/FTorch/page/installation/gpu.html).


## Large tensor support

If your code uses large tensors (where large means more than 2,147,483,647 elements
in any one dimension (the maximum value of a 32-bit integer)), you may
need to compile `ftorch` with 64-bit integers. For information on how to do
this, please see our
[FAQ](https://cambridge-iccs.github.io/FTorch/page/troubleshooting.html#faq)

## Examples

Examples of how to use this library are provided in the [examples directory](examples/).  
They demonstrate different functionalities of the code and are provided with
instructions to modify, build, and run as necessary.

## Tests

For information on testing, see the corresponding
[webpage](https://cambridge-iccs.github.io/FTorch/page/testing.html)
or the [`README` in the `test` subdirectory](test/README.md).

## License

Copyright &copy; ICCS

*FTorch* is distributed under the [MIT Licence](https://github.com/Cambridge-ICCS/FTorch/blob/main/LICENSE).


## Contributions

Contributions and collaborations are welcome.

For bugs, feature requests, and clear suggestions for improvement please
[open an issue](https://github.com/Cambridge-ICCS/FTorch/issues).

If you have built something upon _FTorch_ that would be useful to others, or can
address an [open issue](https://github.com/Cambridge-ICCS/FTorch/issues), please
[fork the repository](https://github.com/Cambridge-ICCS/FTorch/fork) and open a
pull request.

Detailed guidelines can be found in the
[online developer documentation](https://cambridge-iccs.github.io/FTorch/page/developer.html).


### Code of Conduct
Everyone participating in the _FTorch_ project, and in particular in the
issue tracker, pull requests, and social media activity, is expected to treat other
people with respect and, more generally, to follow the guidelines articulated in the
[Python Community Code of Conduct](https://www.python.org/psf/codeofconduct/).


## Authors and Acknowledgment

*FTorch* is written and maintained by the [ICCS](https://github.com/Cambridge-ICCS)

To cite FTorch in research please refer to:

Atkinson et al., (2025). FTorch: a library for coupling PyTorch models to Fortran.
_Journal of Open Source Software_, 10(107), 7602, [https://doi.org/10.21105/joss.07602](https://doi.org/10.21105/joss.07602)

See the `CITATION.cff` file or click 'Cite this repository' on the right.

See [Contributors](https://github.com/Cambridge-ICCS/FTorch/graphs/contributors)
for a full list of contributors.


## Used by
The following projects make use of this code or derivatives in some way:

* [DataWave CAM-GW](https://github.com/DataWaveProject/CAM/)
* [DataWave - MiMA ML](https://github.com/DataWaveProject/MiMA-machine-learning)\
  See Mansfield and Sheshadri (2024) - [DOI: 10.1029/2024MS004292](https://doi.org/10.1029/2024MS004292)
* [Convection parameterisations in ICON](https://github.com/EyringMLClimateGroup/heuer23_ml_convection_parameterization)\
  See Heuer et al. (2024) - [DOI: 10.1029/2024MS004398](https://doi.org/10.1029/2024MS004398)
* To replace a BiCGStab bottleneck in the GloSea6 Seasonal Forecasting model\
  See Park and Chung (2025) - [DOI: 10.3390/atmos16010060](https://doi.org/10.3390/atmos16010060)
* Emulation of cloud resolving models to reduce computational cost in E3SM\
  See Hu et al. (2025) - [DOI: 10.1029/2024MS004618](https://doi.org/10.1029%2F2024MS004618) (and [code](https://github.com/zyhu-hu/E3SM_nvlab/tree/ftorch/climsim_scripts/perlmutter_scripts))

Are we missing anyone? Let us know.
