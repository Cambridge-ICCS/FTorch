# Fortran-PyTorch-lib

![GitHub](https://img.shields.io/github/license/Cambridge-ICCS/fortran-pytorch-lib)

Code and examples for directly calling Pytorch ML models from Fortran.


## Description

It is desirable be able to run machine learning (ML) models directly in Fortran.
Such models are often trained in some other language (say Python) using popular frameworks (say PyTorch) and saved.
We want to run inference on this model without having to call a Python executable.
To achieve this we use the existing ML C++ interface.

This project provides a library enabling a user to directly couple their PyTorch models to Fortran code.
We provide installation instructions for the libaray as well as instructions and examples for performing coupling.


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
2. Navigate into the source directory by running:  
    ```
    cd fortran-pytorch-lib/fortran-pytorch-lib/
    ```
3. Create a `build` directory and execute cmake from within it using the relevant flags:  
    ```
    mkdir build
    cd build
    cmake .. -DTorch_DIR=</path/to/torch>/share/cmake/Torch
    ```  
    where `</path/to/torch/>` is the path to the libtorch installation - if installed via PyTorch in a venv this will be located at `venv/lib/python3.xx/site-packages/torch`.
    The following CMake flags are available and can be passed as arguments through `-D<Option>=<value>`
    | Option                               | Value                       | Description                                                   |
    | -------------------------------------| --------------------------- | --------------------------------------------------------------|
    | `CMAKE_Fortran_COMPILER`             | `ifort` / `gfortran`        | Specify a Fortran compiler to build the library with          |
    | `CMAKE_C_COMPILER`                   | `icc` / `gcc`               | Specify a C compiler to build the library with                |
    | `CMAKE_CXX_COMPILER`                 | `icc` / `gcc`               | Specify a C++ compiler to build the library with              |
    | `CMAKE_INSTALL_PREFIX`               | `</path/to/install/lib/at/>`| Location at which the library files should be installed       |
4. Make and install the code to the chosen location with:
    ```
    make
    make install
    ```  
    This will place the following directories at `</path/to/install/lib/at/>`:  
    * `bin/` - contains example executables
    * `include/` - contains header and mod files
    * `lib64/` - contains cmake and `.so` files


## Usage

TODO How to link to the library from other code.


## License

Copyright &copy; ICCS

*Fortran-PyTorch-Lib* is distributed under the [MIT Licence](https://github.com/Cambridge-ICCS/fortran-pytorch-lib/blob/main/LICENSE).


## Contributions

Contributions and collaborations are welcome.

For bugs, feature requests, and clear suggestions for improvement please
[open an issue](https://github.com/Cambridge-ICCS/fortran-pytorch-lib/issues).

If you have built something upon _Fortran-PyTorchLib_ that would be useful to others, or can
address an [open issue](https://github.com/Cambridge-ICCS/fortran-pytorch-lib/issues), please
[fork the repository](https://github.com/Cambridge-ICCS/fortran-pytorch-lib/fork) and open a
pull request.


### Code of Conduct
Everyone participating in the _Fortran-PyTorch-Lib_ project, and in particular in the
issue tracker, pull requests, and social media activity, is expected to treat other
people with respect and more generally to follow the guidelines articulated in the
[Python Community Code of Conduct](https://www.python.org/psf/codeofconduct/).


## Authors and Acknowledgment

*Fortran-PyTorch-Lib* is written and maintained by the [ICCS](https://github.com/Cambridge-ICCS)

Notable contributors to this project are:

* [**@athelaf**](https://github.com/athelaf)
* [**@jatkinson1000**](https://github.com/jatkinson1000)
* [**@SimonClifford**](https://github.com/SimonClifford)

See [Contributors](https://github.com/Cambridge-ICCS/fortran-pytorch-lib/graphs/contributors)
for a full list.


### Used by
The following projects make use of this code or derivatives in some way:

* [DataWave - MiMA ML](https://github.com/DataWaveProject/MiMA-machine-learning)

Are we missing anyone? Let us know.



