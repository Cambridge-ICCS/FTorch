title: Installation and Build
author: Jack Atkinson
date: Last Updated: October 2025

## Installation and Build

Installation of FTorch is performed using the CMake build system.
This is controlled by the `CMakeLists.txt` file in the root FTorch directory.

- [Dependencies](#dependencies)
- [Basic Installation Instructions](#basic-installation-instructions)
- [CMake Build Options](#cmake-build-options)
- [Building FTorch as a Shared vs. Static library](#building-ftorch-as-a-shared-vs-static-library)


### Dependencies

To install FTorch requires the following to be installed on the system:

- [CMake](https://cmake.org/) >= 3.15
- Fortran (2008 standard compliant), C++ (must fully support C++17), and C compilers
- [LibTorch](https://pytorch.org/cppdocs/installing.html)[^1] or [PyTorch](https://pytorch.org/)

[^1]: 
    _The minimal example provided downloads the CPU-only Linux Nightly binary.
    [Alternative versions](https://pytorch.org/get-started/locally/) to match hardware
    may be required._

#### Additional dependencies of the test suite

FTorch's test suite has some additional dependencies.

- You will also need to install the unit testing framework
  [pFUnit](https://github.com/Goddard-Fortran-Ecosystem/pFUnit).
- FTorch's test suite requires that [PyTorch](https://pytorch.org/) has been
  installed, as opposed to LibTorch. We recommend installing `torchvision` in
  the same command (e.g., `pip install torch torchvision`).[^2] Doing so
  ensures that `torch` and `torchvision` are configured in the same way.
- Other Python modules are installed automatically upon building the tests.

[^2]: _For more details, see [here](https://pytorch.org/get-started/locally/)._


### Basic Installation Instructions

The following instructions assume a Unix system.
For installation on Windows, Apple Silicon, Conda Environments, or Codespaces refer to
the [system-specific guidance](|page|installation/systems.html) as the process may differ.

To build the library, first clone it from GitHub to your local machine using either ssh:
```bash
git clone git@github.com:Cambridge-ICCS/FTorch.git
```
or https:
```bash
git clone https://github.com/Cambridge-ICCS/FTorch.git
```

Then navigate to the FTorch directory and create a build directory:
```bash
cd FTorch
mkdir build
cd build
```

From here invoke CMake with the `Release` build type option, plus any other options as
required from the [table below](#cmake-build-options).
Note: you will likely _need_ to provide at least the `CMAKE_PREFIX_PATH` flag
plus any other options to enforce a consistent build on your machine:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

Finally build and install the library using:
```bash
cmake --build . --target install
```
or, if you want to separate these steps:
```bash
cmake --build .
cmake --install .
```

Note: _If using a machine capable of running multiple jobs this can be sped up by
adding `--parallel [<jobs>]` or `-j [<jobs>]` to the cmake build command.
See the [CMake documentation](https://cmake.org/cmake/help/latest/manual/cmake.1.html#cmdoption-cmake-build-j)
for more information._

Installation will place the following directories at the install location:

* `include/` - contains C header and Fortran mod files
* `lib/` - contains `cmake` directory and `.so`/`.a` files  
  Note: _Depending on your system and architecture `lib` may be `lib64`, and 
  you may have `.dll` files or similar._


### CMake build options

The following CMake flags are available and can be passed as arguments through `-D<Option>=<Value>`
in order to tailor your build:

| Option                                                                                            | Value                                   | Description                                                                                                                                                                |
| ------------------------------------------------------------------------------------------------- | ----------------------------            | --------------------------------------------------------------                                                                                                             |
| [`CMAKE_Fortran_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html) | `gfortran` / `ifx` / `ifort`            | Specify a Fortran compiler to build the library with. This should match the Fortran compiler you're using to build the code you are calling this library from.[^3] |
| [`CMAKE_C_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html)       | `gcc` / `icx` / `icc`                   | Specify a C compiler to build the library with.[^3]                                                                                                                |
| [`CMAKE_CXX_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html)     | `g++` / `icx` / `icpc`                  | Specify a C++ compiler to build the library with.[^3]                                                                                                              |
| [`CMAKE_PREFIX_PATH`](https://cmake.org/cmake/help/latest/variable/CMAKE_PREFIX_PATH.html)        | `</path/to/libTorch/>`                  | Location of Torch installation[^4]                                                                                                                                 |
| [`CMAKE_INSTALL_PREFIX`](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html)  | `</path/to/install/lib/at/>`            | Location at which the library files should be installed. By default this is `/usr/local`                                                                                   |
| [`CMAKE_BUILD_TYPE`](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)          | `Release` / `Debug`                     | Specifies build type. The default is `Debug`, use `Release` for production code                                                                                            |
| `CMAKE_BUILD_TESTS`                                                                               | `TRUE` / `FALSE`                        | Specifies whether to compile FTorch's [test suite](|page|/developer/testing.html) as part of the build.[^5]                                                                                     |
| `GPU_DEVICE`                                                                                      | `NONE` / `CUDA` / `HIP` / `XPU` / `MPS` | Specifies the target GPU architecture (if any)[^6]                                                                                                                |
| `MULTI_GPU`                                                                                      | `ON` / `OFF` | Specifies whether to build the tests that involve multiple GPU devices (`ON` by default if `CMAKE_BUILD_TESTS` and `GPU_DEVICE` are set).                                                                                                                |


[^3]:
    _This may need to be the full path to the compiler if CMake
    cannot locate it by default._

[^4]:
    _The path to the Torch installation needs to allow CMake to locate the relevant Torch CMake files.  
    If Torch has been [installed as LibTorch](https://pytorch.org/cppdocs/installing.html)
    then this should be the absolute path to the unzipped LibTorch distribution.
    If Torch has been installed as PyTorch in a Python [venv (virtual environment)](https://docs.python.org/3/library/venv.html),
    e.g. with `pip install torch`, then this should be  
    `</path/to/venv/>lib/python<3.xx>/site-packages/torch/`.  
    You can find the location of your torch install by importing torch from your Python environment (`import torch`) and running `print(torch.__file__)`_

[^5]:
    _To run the tests, your system's MPI must support `use mpi_f08`.  
    Note that `OpenMPI < v2.0` and `MPICH < v3.1` do not support this module._

[^6]:
    _This must match the installed PyTorch/Libtorch library. When installing with pip, the `index-url` flag can be used to ensure a CPU-only or GPU-enabled version is installed, e.g.
    ```
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```
    URLs for alternative versions can be found [here](https://pytorch.org/get-started/locally/)._


For example, to build using the GNU compilers and install to `$HOME/FTorchbin/`
we would need to run:
```bash
cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_Fortran_COMPILER=gfortran \
-DCMAKE_C_COMPILER=gcc \
-DCMAKE_CXX_COMPILER=g++ \
-DCMAKE_PREFIX_PATH=/path/to/venv/lib/python3.xx/site-packages/torch/ \
-DCMAKE_INSTALL_PREFIX=~/FTorchbin
```

Once this completes you should be able to generate the code and install using:
```bash
cmake --build . --target install
```


### Building FTorch as a Shared vs. Static library

FTorch can be built as either a shared or static library depending
on how you want to link it with your own application.

By default, FTorch builds as a shared library:
```bash
cmake -DBUILD_SHARED_LIBS=ON ...
```
This configuration dynamically links FTorch with its dependencies (including
LibTorch) at runtime. A shared build is recommended for *most* users because:

- Multiple programs can use the same FTorch installation.
- You can update FTorch without recompiling dependent executables.

If you prefer to include FTorch directly inside your executable, you can build
it statically:
```bash
cmake -DBUILD_SHARED_LIBS=OFF ...
```

A static build links all FTorch code directly into your application executable.
This can be useful when:

- You want a single self-contained executable.
- You are installing FTorch on an HPC system and intend to use it in an
application for which you want maximum reproducibility (i.e., the FTorch
version embedded in your application is "frozen").

To this second point on building FTorch as a static library, a brief
justification on applications for which this may be relevant is covered
[here](https://github.com/Cambridge-ICCS/FTorch/pull/448#issue-3544429539).

For more general details on shared and static libraries as well as their
trade-offs, see [shared vs. static
libraries](https://medium.com/@mohitk3000/c-libraries-unpacked-shared-libraries-vs-static-libraries-44764b85056a),
[a case for static
linking](https://ro-che.info/articles/2016-09-09-static-binaries-scientific-computing),
and [static linking considered
harmful](https://www.akkadia.org/drepper/no_static_linking.html)


@note
For discussion on how to build and link another code to the FTorch library see the
[generic usage example](|page|/usage/generic_example.html), and the detailed discussion
on the [HPC page](|page|/installation/hpc.html).
@endnote
