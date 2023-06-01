# Fortran-PyTorch-lib

![GitHub](https://img.shields.io/github/license/Cambridge-ICCS/fortran-pytorch-lib)

Code and examples for directly calling Pytorch ML models from Fortran.


## Description

It is desirable be able to run machine learning (ML) models directly in Fortran.
Such models are often trained in some other language (say Python) using popular frameworks (say PyTorch) and saved.
We want to run inference on this model without having to call a Python executable.
To achieve this we use the existing ML C++ interface.

This project provides a library enabling a user to directly couple their PyTorch models to Fortran code.
We provide installation instructions for the library as well as instructions and examples for performing coupling.

Project status: This project is currently in pre-release with documentation and code being prepared for a first release.
As such there may breaking changes made.
If you are interested in using this libratu please get in touch.


## Installation

### Dependencies

To install the library requires the following to be installed on the system:

* cmake >= 3.1
* [libtorch](https://pytorch.org/cppdocs/installing.html) or [PyTorch](https://pytorch.org/)
* Fortran, C++, and C compilers

### Library installation

To build and install the library:

1. Navigate to the location in which you wish to install the source and run:  
    ```
    git clone git@github.com:Cambridge-ICCS/fortran-pytorch-lib.git
    ```
    to clone via ssh, or  
    ```
    git clone https://github.com/Cambridge-ICCS/fortran-pytorch-lib.git
    ```
    to clone via https.  
2. Navigate into the library source directory by running:  
    ```
    cd fortran-pytorch-lib/fortran-pytorch-lib/
    ```
3. Create a `build` directory and execute cmake from within it using the relevant flags:  
    ```
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    ```
    It is likely that you will need to provide the `CMAKE_PREFIX_PATH` flag (see below).  
    The following CMake flags are available and can be passed as arguments through `-D<Option>=<Value>`:
    | Option                                                                                            | Value                        | Description                                                   |
    | ------------------------------------------------------------------------------------------------- | ---------------------------- | --------------------------------------------------------------|
    | [`CMAKE_Fortran_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html) | `ifort` / `gfortran`         | Specify a Fortran compiler to build the library with          |
    | [`CMAKE_C_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html)       | `icc` / `gcc`                | Specify a C compiler to build the library with                |
    | [`CMAKE_CXX_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html)     | `icc` / `gcc`                | Specify a C++ compiler to build the library with              |
    | [`CMAKE_PREFIX_PATH`](https://cmake.org/cmake/help/latest/variable/CMAKE_PREFIX_PATH.html)        | `</path/to/libTorch/>`          | Location of Torch installation<sup>1</sup> |
    | [`CMAKE_INSTALL_PREFIX`](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html)  | `</path/to/install/lib/at/>` | Location at which the library files should be installed. By default this is `/usr/local` |
    | [`CMAKE_BUILD_TYPE`](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)          | `Release` / `Debug`          | Specifies build type.                                         |

    <sup>1</sup> _The path to the Torch installation needs to allow cmake to loate the relevant Torch cmake files.  
          If Torch has been [installed as libtorch](https://pytorch.org/cppdocs/installing.html) then this should be the absolute path to the unzipped libtorch distribution.
          If Torch has been installed as pyTorch in a python [venv (virtual environment)](https://docs.python.org/3/library/venv.html) then this should be `</path/to/venv/>lib/python<3.xx>/site-packages/torch/`_
4. Make and install the code to the chosen location with:
    ```
    make
    make install
    ```
    This will place the following directories at the install location:  
    * `bin/` - contains example executables
    * `include/` - contains header and mod files
    * `lib64/` - contains cmake and `.so` files


## Usage

TODO How to link to the library from other code.
### CMake
Find FTorch, link to executable
```
find_package(FTorch)
target_link_libraries( <executable> PRIVATE FTorch::ftorch )
message(STATUS "Building with Fortran PyTorch coupling")
```

### make
Something like:
```
-I<path/to/library>
```
where?


## Examples


## License

Copyright &copy; ICCS

*Fortran-PyTorch-Lib* is distributed under the [MIT Licence](https://github.com/Cambridge-ICCS/fortran-pytorch-lib/blob/main/LICENSE).


## Contributions

Contributions and collaborations are welcome.

For bugs, feature requests, and clear suggestions for improvement please
[open an issue](https://github.com/Cambridge-ICCS/fortran-pytorch-lib/issues).

If you have built something upon _Fortran-PyTorch-Lib_ that would be useful to others, or can
address an [open issue](https://github.com/Cambridge-ICCS/fortran-pytorch-lib/issues), please
[fork the repository](https://github.com/Cambridge-ICCS/fortran-pytorch-lib/fork) and open a
pull request.


### Code of Conduct
Everyone participating in the _Fortran-PyTorch-Lib_ project, and in particular in the
issue tracker, pull requests, and social media activity, is expected to treat other
people with respect and, more generally, to follow the guidelines articulated in the
[Python Community Code of Conduct](https://www.python.org/psf/codeofconduct/).


## Authors and Acknowledgment

*Fortran-PyTorch-Lib* is written and maintained by the [ICCS](https://github.com/Cambridge-ICCS)

Notable contributors to this project are:

* [**@athelaf**](https://github.com/athelaf)
* [**@jatkinson1000**](https://github.com/jatkinson1000)
* [**@SimonClifford**](https://github.com/SimonClifford)

See [Contributors](https://github.com/Cambridge-ICCS/fortran-pytorch-lib/graphs/contributors)
for a full list.


## Used by
The following projects make use of this code or derivatives in some way:

* [DataWave - MiMA ML](https://github.com/DataWaveProject/MiMA-machine-learning)

Are we missing anyone? Let us know.



