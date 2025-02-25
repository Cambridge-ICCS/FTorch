# FTorch

**_A library for coupling (Py)Torch machine learning models to Fortran_**

![GitHub](https://img.shields.io/github/license/Cambridge-ICCS/FTorch)
![Fortran](https://img.shields.io/badge/Fortran-2008-purple)

This repository contains code, utilities, and examples for directly calling PyTorch ML
models from Fortran.

For full API and user documentation please see the
[online documentation](https://cambridge-iccs.github.io/FTorch/) which is 
significantly more detailed than this README.


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

* Coupling Machine Learning to Numerical (Climate) Models\
  Platform for Advanced Scientific Computing, Zurich - June 2024\
  [Slides](https://jackatkinson.net/slides/PASC24)
* Blending Machine Learning and Numerical Simulation, with Applications to Climate Modelling\
  Durham HPC days, Durham - May 2024\
  [Slides](https://jackatkinson.net/slides/HPC_Durham_2024)
* Reducing the overheads for coupling PyTorch machine learning models to Fortran\
  ML & DL Seminars, LSCE, IPSL, Paris - November 2023\
  [Slides](https://jackatkinson.net/slides/IPSL_FTorch) - [Recording](https://www.youtube.com/watch?v=-NJGuV6Rz6U)
* Reducing the Overhead of Coupled Machine Learning Models between Python and Fortran\
  RSECon23, Swansea - September 2023\
  [Slides](https://jackatkinson.net/slides/RSECon23) - [Recording](https://www.youtube.com/watch?v=Ei6H_BoQ7g4&list=PL27mQJy8eDHmibt_aL3M68x-4gnXpxvZP&index=33)

Project status: This project is currently in pre-release with documentation and code
being prepared for a first release.
As we stabilise the API in preparation for first release there may be some breaking changes.
Please see 
[online updates documentation](https://cambridge-iccs.github.io/FTorch/page/updates.html)
for clear guidance on how to easily update your older code to run with the latest version.\
If you are interested in using this library please get in touch.

_For a similar approach to calling TensorFlow models from Fortran please see [Fortran-TF-lib](https://github.com/Cambridge-ICCS/fortran-tf-lib)._

## Installation

### Dependencies

To install the library requires the following to be installed on the system:

* CMake >= 3.15
* [LibTorch](https://pytorch.org/cppdocs/installing.html)<sup>*</sup> or [PyTorch](https://pytorch.org/)
* Fortran (2008 standard compliant), C++ (must fully support C++17), and C compilers

<sup>*</sup> _The minimal example provided downloads the CPU-only Linux Nightly binary. [Alternative versions](https://pytorch.org/get-started/locally/) may be required._

#### Additional dependencies of the test suite

FTorch's test suite has some additional dependencies.

* You will also need to install the unit testing framework
  [pFUnit](https://github.com/Goddard-Fortran-Ecosystem/pFUnit).
* FTorch's test suite requires that [PyTorch](https://pytorch.org/) has been
  installed, as opposed to LibTorch. We recommend installing `torchvision` in
  the same command (e.g., `pip install torch torchvision`)<sup>*</sup>. Doing so
  ensures that `torch` and `torchvision` are configured in the same way.
* Other Python modules are installed automatically by the `run_test_suite.sh`
  test script (or `run_test_suite.bat` on Windows).


<sup>*</sup> _For more details, see [here](https://pytorch.org/get-started/locally/)._

#### Windows Support

If building in a Windows environment then you can either use
[Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/) (WSL)
or Visual Studio and the Intel Fortran Compiler.
For full details on the process see the
[online Windows documentation](https://cambridge-iccs.github.io/FTorch/page/troubleshooting.html#windows).\
Note that LibTorch is not supported for the GNU Fortran compiler with MinGW.

#### Apple Silicon Support

At the time of writing [there are issues](https://github.com/pytorch/pytorch/issues/143571)
building FTorch on Apple Silicon when linking to downloaded `LibTorch` binaries or
pip-installed PyTorch.
FTorch can successfully be built, including utilising the MPS backend, from inside a
conda environment using the environment files and instructions in
[`conda/`](https://github.com/Cambridge-ICCS/FTorch/tree/main/conda).

#### Conda Support

Conda is not our preferred approach for managing dependencies, but for users who want
an environment to build FTorch in we provide guidance and environment files in
[`conda/`](https://github.com/Cambridge-ICCS/FTorch/tree/main/conda). Note that these
are not minimal and will install Python, PyTorch, and modules required for running the
tests and examples.

### Library installation

For detailed installation instructions please see the
[online installation documentation](https://cambridge-iccs.github.io/FTorch/page/cmake.html).

To build and install the library:

1. Navigate to the location in which you wish to install the source and run:  
    ```
    git clone git@github.com:Cambridge-ICCS/FTorch.git
    ```
    to clone via ssh, or  
    ```
    git clone https://github.com/Cambridge-ICCS/FTorch.git
    ```
    to clone via https.  
2. Navigate to the root FTorch directory by running:  
    ```
    cd FTorch/
    ```
3. Build the library using CMake with the relevant options from the table below:  
    ```
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    ```

    The following table of CMake options are available to be passed as arguments to `cmake` through `-D<Option>=<Value>`.  
    It is likely that you will need to provide at least `CMAKE_PREFIX_PATH`.  
    | Option                                                                                            | Value                        | Description                                                   |
    | ------------------------------------------------------------------------------------------------- | ---------------------------- | --------------------------------------------------------------|
    | [`CMAKE_Fortran_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html) | `gfortran` / `ifx` / `ifort` | Specify a Fortran compiler to build the library with. This should match the Fortran compiler you're using to build the code you are calling this library from.<sup>1</sup>        |
    | [`CMAKE_C_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html)       | `gcc` / `icx` / `icc`        | Specify a C compiler to build the library with.<sup>1</sup>                |
    | [`CMAKE_CXX_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html)     | `g++` / `icx` / `icpc`       | Specify a C++ compiler to build the library with.<sup>1</sup>              |
    | [`CMAKE_PREFIX_PATH`](https://cmake.org/cmake/help/latest/variable/CMAKE_PREFIX_PATH.html)        | `</path/to/LibTorch/>`       | Location of Torch installation<sup>2</sup>                    |
    | [`CMAKE_INSTALL_PREFIX`](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html)  | `</path/to/install/lib/at/>` | Location at which the library files should be installed. By default this is `/usr/local` |
    | [`CMAKE_BUILD_TYPE`](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)          | `Release` / `Debug`          | Specifies build type. The default is `Debug`, use `Release` for production code|
    | `CMAKE_BUILD_TESTS`                                                                               | `TRUE` / `FALSE`             | Specifies whether to compile FTorch's [test suite](https://cambridge-iccs.github.io/FTorch/page/testing.html) as part of the build. |
    | `GPU_DEVICE` | `NONE` / `CUDA` / `XPU` / `MPS` | Specifies the target GPU architecture (if any) <sup>3</sup> |

    <sup>1</sup> _On Windows this may need to be the full path to the compiler if CMake cannot locate it by default._  

    <sup>2</sup> _The path to the Torch installation needs to allow CMake to locate the relevant Torch CMake files.  
          If Torch has been [installed as LibTorch](https://pytorch.org/cppdocs/installing.html)
          then this should be the absolute path to the unzipped LibTorch distribution.
          If Torch has been installed as PyTorch in a Python [venv (virtual environment)](https://docs.python.org/3/library/venv.html),
          e.g. with `pip install torch`, then this should be `</path/to/venv/>lib/python<3.xx>/site-packages/torch/`.  
		  You can find the location of your torch install by importing torch from your Python environment (`import torch`) and running `print(torch.__file__)`_

    <sup>3</sup> _This is often overridden by PyTorch. When installing with pip, the `index-url` flag can be used to ensure a CPU-only or GPU-enabled version is installed, e.g.
          `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
          URLs for alternative versions can be found [here](https://pytorch.org/get-started/locally/)._

4. Make and install the library to the desired location with either:
	```
    cmake --build . --target install
    ```
    or, if you want to separate these steps:
    ```
    cmake --build .
    cmake --install .
    ```

   Note: If using a machine capable of running multiple jobs this can be sped up by
   adding `--parallel [<jobs>]` or `-j [<jobs>]` to the `cmake build` command.
   See the [CMake documentation](https://cmake.org/cmake/help/latest/manual/cmake.1.html#cmdoption-cmake-build-j)
   for more information.

    Installation will place the following directories at the install location:  
    * `CMAKE_INSTALL_PREFIX/include/` - contains header and mod files
    * `CMAKE_INSTALL_PREFIX/lib/` - contains `cmake` directory and `.so` files  
    _Note: depending on your system and architecture `lib` may be `lib64`, and 
    you may have `.dll` files or similar._  
	_Note: In a Windows environment this will require administrator privileges for the default install location._


## Usage

In order to use FTorch users will typically need to follow these steps:

1. Save a PyTorch model as [TorchScript](https://pytorch.org/docs/stable/jit.html).
2. Write Fortran using the FTorch bindings to use the model from within Fortran.
3. Build and compile the code, linking against the FTorch library

These steps are described in more detail in the
[online documentation](https://cambridge-iccs.github.io/FTorch/page/examples.html#overview-of-the-interfacing-process)


## GPU Support

To run on GPU requires an installation of LibTorch compatible for the GPU device
you wish to target and two main adaptations to the code:

1. When saving a TorchScript model, ensure that it is on the appropriate GPU
   device type. The `pt2ts.py` script has a command line argument
   `--device_type`, which currently accepts four different device types: `cpu`
   (default), `cuda`, `xpu`, or `mps`.
2. When using FTorch in Fortran, set the device for the input
   tensor(s) to the appropriate GPU device type, rather than `torch_kCPU`. There
   are currently three options: `torch_kCUDA`, `torch_kXPU`, or `torch_kMPS`.

For detailed guidance about running on GPU, including instructions for using multiple
devices, please see the
[online GPU documentation](https://cambridge-iccs.github.io/FTorch/page/gpu.html).

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

Notable contributors to this project are:

* [**@athelaf**](https://github.com/athelaf)
* [**@jatkinson1000**](https://github.com/jatkinson1000)
* [**@SimonClifford**](https://github.com/SimonClifford)
* [**@ElliottKasoar**](https://github.com/ElliottKasoar)
* [**@TomMelt**](https://github.com/TomMelt)
* [**@jwallwork23**](https://github.com/jwallwork23)

See [Contributors](https://github.com/Cambridge-ICCS/FTorch/graphs/contributors)
for a full list.


## Used by
The following projects make use of this code or derivatives in some way:

* [M2LInES CAM-ML](https://github.com/m2lines/CAM-ML)
* [DataWave CAM-GW](https://github.com/DataWaveProject/CAM/)
* [DataWave - MiMA ML](https://github.com/DataWaveProject/MiMA-machine-learning)\
  See Mansfield and Sheshadri (2024) - [DOI: 10.1029/2024MS004292](https://doi.org/10.1029/2024MS004292)
* [Convection parameterisations in ICON](https://github.com/EyringMLClimateGroup/heuer23_ml_convection_parameterization)\
  See Heuer et al. (2023) - [DOI: 10.48550/arXiv.2311.03251](https://doi.org/10.48550/arXiv.2311.03251)

Are we missing anyone? Let us know.
