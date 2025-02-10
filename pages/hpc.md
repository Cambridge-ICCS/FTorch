title: Guidance for use in High Performance Computing (HPC)

[TOC]

A common application of FTorch (indeed, the driving one for development) is the
coupling of machine learning components to models running on HPC systems.

Here we provide some guidance/hints to help with deploymwent in these settings.

## Installation

### Building for basic use

The basic installation procedure is the same as described in the
[main documentation](page//cmake.html) and README, cloning from
[GitHub](https://github.com/Cambridge-ICCS/FTorch) and building using CMake.

### Obtaining LibTorch

For use on a HPC system we advise linking to an installation of LibTorch rather than
installing full PyTorch.
This will reduce the dependencies and remove any requirement of Python.
LibTorch can be obtained from the
[PyTorch website](https://pytorch.org/get-started/locally/).

Once you have successfully tested and deployed FTorch in your code we recommend speaking
to your administrator/software stack manager to make your chosen version of libtorch
loadable as a `module`.
This will improve reproducibility and simplify the process for future users on your
system.
See the [information below](#libtorch-as-a-module) for further details.

### Environment management

It is important that FTorch is built using the same environment and compilers as the
software to which it will be linked.

Therefore before running the build you should ensure that you match the environment to
that which your code will be built with.
This will usually be done by using the same `module` commands as you would use to build
the model:
```
module purge
module load ...
```

Alternatively you may be provided with a shell script that runs these commands and sets
environment variables etc. that can be sourced:
```
source model_environment.sh
```

Complex models with custom build systems may obfuscate this process, and you might need
to probe the build system/scripts for this information.
If in doubt speak to the maintainer of the software for your system, or the manager of
the software stack on the machine.

Because of the need to match compilers it is strongly recommended to use the
`CMAKE_Fortran_COMPILER`, `CMAKE_C_COMPILER`, and `CMAKE_CXX_COMPILER` when building
with CMake to enforce this.

### Building Projects and Linking to FTorch

Whilst we describe how to link to FTorch using CMake to build a project on our main
page, many HPC models do not use CMake and rely on `make` or more elaborate build
systems.
To build a project with `make` or similar you need to _include_ the FTorch library when
compiling and _link_ the executable against it.

To compile with make add the following compiler flag when compiling files that
use ftorch to _include_ the library:
```
-I<path/to/FTorch/install/location>/include/ftorch
```
This is often done by appending to an `FCFLAGS` compiler flags variable or similar:
```
FCFLAGS += -I<path/to/FTorch/install/location>/include/ftorch
```

When compiling the final executable add the following link flag:
```
-L<path/to/FTorch/install/location>/lib64 -lftorch
```
This is often done by appending to an `LDFLAGS` linker flags variable or similar:
```
LDFLAGS += -L<path/to/FTorch/install/location>/lib64 -lftorch
```

You may also need to add the location of the dynamic library `.so` files to your
`LD_LIBRARY_PATH` environment variable unless installing in a default location:
```
export LD_LIBRARY_PATH = $LD_LIBRARY_PATH:<path/to/FTorch/installation>/lib64
```

> Note: _Depending on your system and architecture `lib` may be `lib64` or something similar._

> Note: _On MacOS devices you will need to set `DYLD_LIBRARY_PATH` rather than `LD_LIBRARY_PATH`._

Whilst experimenting it may be useful to build FTorch using the `CMAKE_BUILD_TYPE=RELEASE`
CMake flag to allow useful error messages and investigation with debugging tools.


### Module systems

Most HPC systems are managed using [Environment Modules](https://modules.sourceforge.net/).
To build FTorch it is important you
[match the environment in which you build FTorch to that of the executable](#environment-management)
by loading the same modules as when building the main code.

As a minimal requirement you will need to load modules for compilers and cmake.
Further functionalities may require loading of additional modules such as an
MPI installation and CUDA.
Some systems may also have pFUnit available ada loadable module to save you needing to
build from scratch per the documentation.

#### LibTorch as a module

Once you have a working build of FTorch it is advisable to pin the version of LibTorch
and make it a loadable module to improve reproducibility and simplify the build process
for subsequent users on the system.

This can be done my the software manager after which you can use
```
module load libtorch
```
or similar instead of downloading the binary from the PyTorch website.

Note that the module name on your system may include additional information about the
version, compilers used, and a hash code.

#### FTorch as a module 

If there are many users who want to use FTorch on a system it may be worth building
and making it loadable as a module itself.
The module should be labelled with the compilers it was built with (see the
[importance of enviroinment matching](#environment-management)) and automatically load
any subdependencies (CUDA)

The build should be completed for `CMAKE_BUILD_TYPE=RELEASE` and run the unit tests to
check successful installation.

Once complete it should be possible to:
```
module load ftorch
```
or similar.

This process should also add FTorch to the `LD_LIBRARY_PATH` and `CMAKE_PREFIX_PATH`
rather than requiring the user to specify them manually as suggested elsewhere in this
documentation.
