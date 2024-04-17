# FTorch

**_A library for coupling (Py)Torch machine learning models to Fortran_**

![GitHub](https://img.shields.io/github/license/Cambridge-ICCS/FTorch)

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
type(torch_module) :: model
type(torch_tensor), dimension(n_inputs) :: model_inputs_arr
type(torch_tensor) :: model_output
...
model = torch_module_load("/my/saved/TorchScript/model.pt")
model_inputs_arr(1) = torch_tensor_from_array(input_fortran, in_layout, torch_kCPU)
model_output = torch_tensor_from_array(output_fortran, out_layout, torch_kCPU)

call torch_module_forward(model, model_input_arr, n_inputs, model_output)
```

The following presentations provide an introduction and overview of _FTorch_:

* Reducing the overheads for coupling PyTorch machine learning models to Fortran\
  ML & DL Seminars, LSCE, IPSL, Paris - November 2023\
  [Slides](https://jackatkinson.net/slides/IPSL_FTorch/IPSL_FTorch.html) - [Recording](https://www.youtube.com/watch?v=-NJGuV6Rz6U)
* Reducing the Overhead of Coupled Machine Learning Models between Python and Fortran\
  RSECon23, Swansea - September 2023\
  [Slides](https://jackatkinson.net/slides/RSECon23/RSECon23.html) - [Recording](https://www.youtube.com/watch?v=Ei6H_BoQ7g4&list=PL27mQJy8eDHmibt_aL3M68x-4gnXpxvZP&index=33)

Project status: This project is currently in pre-release with documentation and code
being prepared for a first release.
As such breaking changes may be made.
If you are interested in using this library please get in touch.

_For a similar approach to calling TensorFlow models from Fortran please see [Fortran-TF-lib](https://github.com/Cambridge-ICCS/fortran-tf-lib)._

## Installation

### Dependencies

To install the library requires the following to be installed on the system:

* CMake >= 3.1
* [libtorch](https://pytorch.org/cppdocs/installing.html)<sup>*</sup> or [PyTorch](https://pytorch.org/)
* Fortran, C++ (must fully support C++17), and C compilers

<sup>*</sup> _The minimal example provided downloads the CPU-only Linux Nightly binary. [Alternative versions](https://pytorch.org/get-started/locally/) may be required._

#### Windows Support

If building in a Windows environment then you can either use
[Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/) (WSL)
or Visual Studio and the Intel Fortran Compiler.
For full details on the process see the
[online Windows documentation](https://cambridge-iccs.github.io/FTorch/page/troubleshooting.html#windows).\
Note that libTorch is not supported for the GNU Fortran compiler with MinGW.

#### Apple Silicon Support

At the time of writing, libtorch is only officially available for x86 architectures
(according to https://pytorch.org/). However, the version of PyTorch provided by
`pip install torch` uses an ARM binary for libtorch which works on Apple Silicon.

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
2. Navigate to the source directory by running:  
    ```
    cd FTorch/src/
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
    | [`CMAKE_Fortran_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html) | `ifort` / `gfortran`         | Specify a Fortran compiler to build the library with. This should match the Fortran compiler you're using to build the code you are calling this library from.<sup>1</sup>        |
    | [`CMAKE_C_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html)       | `icc` / `gcc`                | Specify a C compiler to build the library with.<sup>1</sup>                |
    | [`CMAKE_CXX_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html)     | `icpc` / `g++`               | Specify a C++ compiler to build the library with.<sup>1</sup>              |
    | [`CMAKE_PREFIX_PATH`](https://cmake.org/cmake/help/latest/variable/CMAKE_PREFIX_PATH.html)        | `</path/to/libTorch/>`       | Location of Torch installation<sup>2</sup>                    |
    | [`CMAKE_INSTALL_PREFIX`](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html)  | `</path/to/install/lib/at/>` | Location at which the library files should be installed. By default this is `/usr/local` |
    | [`CMAKE_BUILD_TYPE`](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)          | `Release` / `Debug`          | Specifies build type. The default is `Debug`, use `Release` for production code|
    | `ENABLE_CUDA`                                                                                     | `TRUE` / `FALSE`             | Specifies whether to check for and enable CUDA<sup>2</sup> |

    <sup>1</sup> _On Windows this may need to be the full path to the compiler if CMake cannot locate it by default._  

    <sup>2</sup> _The path to the Torch installation needs to allow CMake to locate the relevant Torch CMake files.  
          If Torch has been [installed as libtorch](https://pytorch.org/cppdocs/installing.html)
          then this should be the absolute path to the unzipped libtorch distribution.
          If Torch has been installed as PyTorch in a Python [venv (virtual environment)](https://docs.python.org/3/library/venv.html),
          e.g. with `pip install torch`, then this should be `</path/to/venv/>lib/python<3.xx>/site-packages/torch/`.  
		  You can find the location of your torch install by importing torch from your Python environment (`import torch`) and running `print(torch.__file__)`_
	  
4. Make and install the library to the desired location with either:
	```
    cmake --build . --target install
    ```
    or, if you want to separate these steps:
    ```
    cmake --build .
    cmake --install .
    ```

    Note: If you are using CMake < 3.15 then you will need to build and install separately
    using the make system specific commands.
    For example, if using `make` on UNIX this would be:
    ```
    make
    make install
    ```

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

To run on GPU requires a CUDA-compatible installation of libtorch and two main
adaptations to the code:

1. When saving a TorchScript model, ensure that it is on the GPU
2. When using FTorch in Fortran, set the device for the input
   tensor(s) to `torch_kCUDA`, rather than `torch_kCPU`.

For detailed guidance about running on GPU, including instructions for using multiple
devices, please see the
[online GPU documentation](https://cambridge-iccs.github.io/FTorch/page/gpu.html).

## Examples

Examples of how to use this library are provided in the [examples directory](examples/).  
They demonstrate different functionalities of the code and are provided with
instructions to modify, build, and run as necessary.


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
[online developer documentation](page/developer.html).


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

See [Contributors](https://github.com/Cambridge-ICCS/FTorch/graphs/contributors)
for a full list.


## Used by
The following projects make use of this code or derivatives in some way:

* [M2LInES CAM-ML](https://github.com/m2lines/CAM-ML)
* [DataWave CAM-GW](https://github.com/DataWaveProject/CAM/)
* [DataWave - MiMA ML](https://github.com/DataWaveProject/MiMA-machine-learning)

Are we missing anyone? Let us know.
