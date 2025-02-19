title: Installation and Build Process

[TOC]

Installation of FTorch is done by CMake.

This is controlled by the `CMakeLists.txt` file in the root FTorch directory.

## Dependencies

To install the library requires the following to be installed on the system:

- CMake >= 3.15
- LibTorch or PyTorch
- Fortran (2008 standard compliant), C++ (must fully support C++17), and C compilers


## Basic instructions

To build the library, first clone it from GitHub to your local machine and then run:
```
cd FTorch/
mkdir build
cd build
```

Then invoke CMake with the `Release` build option, plus any other options as required
from the table below in [CMake build options](#cmake-build-options)
(note: you will likely _need_ to add some of these options to enforce a consistent
build on your machine):
```
cmake .. -DCMAKE_BUILD_TYPE=Release
```

Finally build and install the library using:
```
cmake --build . --target install
```
or, if you want to separate these steps:
```
cmake --build .
cmake --install .
```

> Note: _if you are building on Windows please refer to the
> [Windows install guidance](page/troubleshooting.html#windows) as the process will
> likely differ from the UNIX-based stsyems covered here._

## CMake build options

### FTorch Library

It is likely that you will need to provide at least the `CMAKE_PREFIX_PATH` flag.  
The following CMake flags are available and can be passed as arguments through `-D<Option>=<Value>`:

| Option                                                                                            | Value                        | Description                                                   |
| ------------------------------------------------------------------------------------------------- | ---------------------------- | --------------------------------------------------------------|
| [`CMAKE_Fortran_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html) | `gfortran` / `ifx` / `ifort` | Specify a Fortran compiler to build the library with. This should match the Fortran compiler you're using to build the code you are calling this library from.<sup>1</sup>        |
| [`CMAKE_C_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html)       | `gcc` / `icx` / `icc`        | Specify a C compiler to build the library with.<sup>1</sup>                |
| [`CMAKE_CXX_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html)     | `g++` / `icx` / `icpc`       | Specify a C++ compiler to build the library with.<sup>1</sup>              |
| [`CMAKE_PREFIX_PATH`](https://cmake.org/cmake/help/latest/variable/CMAKE_PREFIX_PATH.html)        | `</path/to/libTorch/>`       | Location of Torch installation<sup>2</sup>                    |
| [`CMAKE_INSTALL_PREFIX`](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html)  | `</path/to/install/lib/at/>` | Location at which the library files should be installed. By default this is `/usr/local` |
| [`CMAKE_BUILD_TYPE`](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)          | `Release` / `Debug`          | Specifies build type. The default is `Debug`, use `Release` for production code|
| `CMAKE_BUILD_TESTS`                                                                               | `TRUE` / `FALSE`             | Specifies whether to compile FTorch's [test suite](testing.html) as part of the build. |
| `GPU_DEVICE` | `NONE` / `CUDA` / `XPU` / `MPS` | Specifies the target GPU architecture (if any) <sup>3</sup> |



> <sup>1</sup> _On Windows this may need to be the full path to the compiler if CMake
> cannot locate it by default._
>
> <sup>2</sup> _The path to the Torch installation needs to allow CMake to locate the relevant Torch CMake files.  
>       If Torch has been [installed as LibTorch](https://pytorch.org/cppdocs/installing.html)
>       then this should be the absolute path to the unzipped LibTorch distribution.
>       If Torch has been installed as PyTorch in a Python [venv (virtual environment)](https://docs.python.org/3/library/venv.html),
>       e.g. with `pip install torch`, then this should be `</path/to/venv/>lib/python<3.xx>/site-packages/torch/`._
>
> <sup>3</sup> _This is often overridden by PyTorch. When installing with pip, the `index-url` flag can be used to ensure a CPU-only or GPU-enabled version is installed, e.g.
>       `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
>       URLs for alternative versions can be found [here](https://pytorch.org/get-started/locally/)._

For example, to build on a unix system using the gnu compilers and install to `$HOME/FTorchbin/`
we would need to run:
```
cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_Fortran_COMPILER=gfortran \
-DCMAKE_C_COMPILER=gcc \
-DCMAKE_CXX_COMPILER=g++ \
-DCMAKE_PREFIX_PATH=/path/to/venv/lib/python3.xx/site-packages/torch/ \
-DCMAKE_INSTALL_PREFIX=~/FTorchbin
```

Once this completes you should be able to generate the code and install using:
```
cmake --build . --target install
```

> Note: _If using a machine capable of running multiple jobs this can be sped up by
> adding `--parallel [<jobs>]` or `-j [<jobs>]` to the `cmake build` command.
> See the [CMake documentation](https://cmake.org/cmake/help/latest/manual/cmake.1.html#cmdoption-cmake-build-j)
> for more information._

Installation will place the following directories at the install location:

* `CMAKE_INSTALL_PREFIX/include/` - contains C header and Fortran mod files
* `CMAKE_INSTALL_PREFIX/lib64/` - contains `cmake` directory and `.so` files

> Note: _In a Windows environment this will require administrator privileges for the default install location._


### Other projects using CMake

We generally advise building projects that make use of FTorch with CMake where possible.

If doing this you need to include the following in the `CMakeLists.txt` file to
find the FTorch installation and link it to the executable.

```CMake
find_package(FTorch)
target_link_libraries( <executable> PRIVATE FTorch::ftorch )
message(STATUS "Building with Fortran PyTorch coupling")
```

You will then need to use the `-DFTorch_DIR=</path/to/install/location>` flag
when running CMake.


## Building other projects with make

To build a project with `make` you need to include the FTorch library when compiling
and link the executable against it.

For full details of the flags to set and the linking process see the
[HPC build pages](page/hpc.html).


## Conda Support

Conda is not our preferred approach for managing dependencies, but for users who want
an environment to build FTorch in we provide [guidance and environment files](https://github.com/Cambridge-ICCS/FTorch/tree/main/conda).
Note that these are not minimal and will install Python, PyTorch, and modules required
for running the tests and examples.
