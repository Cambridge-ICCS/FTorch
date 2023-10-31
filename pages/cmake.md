title: Installation and Build Process

[TOC]

Installation of FTorch is done by CMake.

This is controlled by the `CMakeLists.txt` file in `src/`.

## Basic instructions

To build the library, first clone it from github to your local machine and then run:
```bash
cd FTorch/src/
mkdir build
cd build
```

Then invoke CMake with the Release build option, plus any other options as required
from the table below in [CMake build options](#cmake-build-options)
(note: you will likely _need_ to add some of these options to ):
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

Finally build and install the library:
```bash
cmake --build . --target install --config Release
```

## CMake build options

### FTorch Library

It is likely that you will need to provide at least the `CMAKE_PREFIX_PATH` flag.  
The following CMake flags are available and can be passed as arguments through `-D<Option>=<Value>`:

| Option                                                                                            | Value                        | Description                                                   |
| ------------------------------------------------------------------------------------------------- | ---------------------------- | --------------------------------------------------------------|
| [`CMAKE_Fortran_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html) | `ifort` / `gfortran`         | Specify a Fortran compiler to build the library with. This should match the Fortran compiler you're using to build the code you are calling this library from.        |
| [`CMAKE_C_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html)       | `icc` / `gcc`                | Specify a C compiler to build the library with                |
| [`CMAKE_CXX_COMPILER`](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html)     | `icpc` / `g++`               | Specify a C++ compiler to build the library with              |
| [`CMAKE_PREFIX_PATH`](https://cmake.org/cmake/help/latest/variable/CMAKE_PREFIX_PATH.html)        | `</path/to/libTorch/>`       | Location of Torch installation<sup>1</sup>                    |
| [`CMAKE_INSTALL_PREFIX`](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html)  | `</path/to/install/lib/at/>` | Location at which the library files should be installed. By default this is `/usr/local` |
| [`CMAKE_BUILD_TYPE`](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)          | `Release` / `Debug`          | Specifies build type. The default is `Debug`, use `Release` for production code|
| `ENABLE_CUDA`                                                                                     | `TRUE` / `FALSE`             | Specifies whether to check for and enable CUDA<sup>2</sup> |


<sup>1</sup> _The path to the Torch installation needs to allow cmake to locate the relevant Torch cmake files.  
      If Torch has been [installed as libtorch](https://pytorch.org/cppdocs/installing.html)
      then this should be the absolute path to the unzipped libtorch distribution.
      If Torch has been installed as PyTorch in a python [venv (virtual environment)](https://docs.python.org/3/library/venv.html),
      e.g. with `pip install torch`, then this should be `</path/to/venv/>lib/python<3.xx>/site-packages/torch/`._

<sup>2</sup> _This is often overridden by PyTorch. When installing with pip, the `index-url` flag can be used to ensure a CPU or GPU only version is installed, e.g.
      `pip install torch --index-url https://download.pytorch.org/whl/cpu`
      or
      `pip install torch --index-url https://download.pytorch.org/whl/cu118`
      (for CUDA 11.8). URLs for alternative versions can be found [here](https://pytorch.org/get-started/locally/)._

For example, to build on a unix system using the gnu compilers and install to `$HOME/FTorchbin/`
we would need to run:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_PREFIX_PATH=/path/to/venv/lib/python3.11/site-packages/torch/ -DCMAKE_INSTALL_PREFIX=~/FTorchbin
```

Once this completes you should be able to generate the code and install using:
```bash
cmake --build . --target install --config Release
```
or, if you want to separate these into two steps:
```bash
cmake --build .
cmake --install . --config Release
```

Note: If you are using cmake<3.15 then you will need to build and install separately
using the make system specific commands.
For example, if using `make` on UNIX this would be:
```bash
make
make install
```

Installation will place the following directories at the install location:

* `CMAKE_INSTALL_PREFIX/include/` - contains C header and Fortran mod files
* `CMAKE_INSTALL_PREFIX/lib64/` - contains `cmake` directory and `.so` files

_Note: In a Windows environment this will require administrator privileges for the default install location._


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
when running cmake.


## Building other projects with make

To build a project with make you need to include the FTorch library when compiling
and link the executable against it.

To compile with make add the following compiler flag when compiling files that
use ftorch:
```
FCFLAGS += -I<path/to/install/location>/include/ftorch
```

When compiling the final executable add the following link flag:
```
LDFLAGS += -L<path/to/install/location>/lib64 -lftorch
```

You may also need to add the location of the `.so` files to your `LD_LIBRARY_PATH`
unless installing in a default location:
```
export LD_LIBRARY_PATH = $LD_LIBRARY_PATH:<path/to/installation>/lib64
```
