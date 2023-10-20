title: CMake Build Process

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
cmake --build . --target install
```

## CMake build options

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
```

```

