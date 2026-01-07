title: HPC Support
author: Jack Atkinson
date: Last Updated: October 2025

# Guidance for use in High Performance Computing (HPC)

- [Installation](#installation)
- [Building and Linking](#building-and-linking)
- [Parallelism](#parallelism)

A common application of FTorch (indeed, the driving one for development) is the
coupling of machine learning components to large scientific codes/models running on
HPC systems.

Here we provide some guidance to help with deployment in these settings.

## Installation

The basic installation procedure is the same as described in the
[main documentation](|page|/installation/general.html) and README, cloning from
[GitHub](https://github.com/Cambridge-ICCS/FTorch) and building using CMake.

### Obtaining LibTorch

For use on a HPC system we advise linking to an installation of LibTorch rather than
installing full PyTorch.
This will reduce the dependencies and remove any requirement of Python.
LibTorch can be obtained from the
[PyTorch website](https://pytorch.org/get-started/locally/).
The assumption here is that any Python/PyTorch development is done elsewhere with a
model being saved to TorchScript for subsequent use by FTorch.

Once you have successfully tested and deployed FTorch in your code we recommend speaking
to your administrator/software stack manager to make your chosen version of libtorch
loadable as a `module`.
This will improve reproducibility and simplify the process for future users on your
system.
See the [information below](#libtorch-as-a-module) for further details.


### Environment management

It is important that FTorch is built using the same environment and compilers as the
model/code in which it will be used.

Most HPC systems manage software environments through
[Environment Modules](https://modules.sourceforge.net/).
It is important to load the same modules when building FTorch as you would
when building your main code.
This will usually be done by using the same `module` commands as you would use to build
your code/model:
```sh
module purge
module load ...
```

Alternatively you may be provided with a shell script that runs these commands and sets
environment variables etc. that can be sourced:
```sh
source model_environment.sh
```

Complex codes with custom build systems may obfuscate this process, and you might need
to probe the build system/scripts for this information.
If in doubt speak to the maintainer of the software for your system, or the manager of
the software stack on the machine.

As a minimal requirement you will need to load modules for compilers and CMake.
These may be installed by the base OS/environment, but it is recommended to use modules
for reproducibility, access to a wider range of versions, and to match the compilers
used to build the main code.
Because of the need to match compilers it is strongly recommended to
[specify the CMake flags](|page|/installation/general.html#cmake-build-options)
`CMAKE_Fortran_COMPILER`, `CMAKE_C_COMPILER`, and `CMAKE_CXX_COMPILER` when building.

Further functionalities may require loading of additional modules such as an
MPI installation and CUDA.


#### LibTorch as a module

Once you have a working build of FTorch it is advisable to pin the version of LibTorch
and make it a loadable module to improve reproducibility and simplify the build process
for subsequent users on the system.

This can be done by the manager of the software stack, after which you can use
```sh
module load libtorch
```
or similar instead of downloading the binary from the PyTorch website.

Note that the module name on your system may include additional information about the
version, compilers used, and a hash code as well as the name `libtorch`.

You should always verify that the version of LibTorch used to build FTorch is
compatible with the version used to build and train your PyTorch model and save to
TorchScript. If a newer version is used in Python it may be neccessary to provide the
matching version in the software stack.


#### FTorch as a module 

If there are many users who want to use FTorch on a system it may be worth building
and making it loadable as a module itself.
The module should be labelled with the compilers it was built with (see the
[importance of environment matching](#environment-management)) and automatically load
any subdependencies (e.g. LibTorch, CUDA)

For production builds, ensure that FTorch is built using the `CMAKE_BUILD_TYPE=Release`
[flag](|page|/installation/general.html#cmake-build-options) to enable optimisations.
It is also recommended to run FTorch's unit tests after building to verify successful
installation.
For details on running FTorch's unit and integration tests after building, see the [testing documentation](|page|/developer/testing.html).

Once complete it should be possible to:
```sh
module load ftorch
```
or similar.
Loading an ftorch module should also add to the `LD_LIBRARY_PATH` and
`CMAKE_PREFIX_PATH`, rather than requiring the user to specify them manually as
discussed in [building and linking](|page|/installation/hpc.html#building-and-linking)
below.


## Building and Linking

Whilst we describe how to link to FTorch using CMake to build a project in our
[generic usage example](|page|/usage/generic_example.html),
many HPC codes rely on `make` or more elaborate custom build systems.
To build a project with `make` or similar you need to _include_ the FTorch's
header (`.h`) and module (`.mod`) files and _link_ the executable
to the Ftorch library (e.g., `.so`, `.dll`, `.dylib` depending on your system) when
compiling.

To compile with `make` use the following compiler flag for any files that
use ftorch to _include_ the module and header files.
This is often done by appending to an `FCFLAGS` compiler flags variable or similar:
```sh
FCFLAGS+=" -I<path/to/FTorch/install/location>/include/ftorch"
```

When compiling the final executable add the following _linker_ flag.
This is often done by appending to an `LDFLAGS` linker flags variable or similar:
```sh
LDFLAGS+=" -L<path/to/FTorch/install/location>/lib -lftorch"
```

### pkg-config

If you have [pkg-config](https://en.wikipedia.org/wiki/Pkg-config) installed,
you can easily query the compiler and linker flags of FTorch instead of specifying
them manually as above.
FTorch provides a standard pkg-config file in the `lib/` installation for this purpose allowing users to instead use:
```sh
FCFLAGS+=" $(pkg-config --cflags </path/to/FTorch/install/location>/lib/pkgconfig/ftorch.pc)"
LDFLAGS+=" $(pkg-config --libs </path/to/FTorch/install/location>/lib/pkgconfig/ftorch.pc)"
```
or, further simplified by extensing teh `PKG_CONFIG_PATH` environment variable:
```sh
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:</path/to/FTorch/install/location>/lib/pkgconfig/
FCFLAGS+=" $(pkg-config --cflags ftorch)"
LDFLAGS+=" $(pkg-config --libs ftorch)"
```

### Adding to the runtime library path

You may also need to add the location of the dynamic library `.so` files to your
`LD_LIBRARY_PATH` environment variable unless installing in a default location:
```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path/to/FTorch/installation>/lib
```

FTorch depends on Torch, and the `RPATH` of the FTorch shared library (i.e.,
`libftorch.so`) generated during CMake installation contains the path to its
Torch shared library dependency. The dynamic linker of the GNU C library
searches for shared libraries first using the `RPATH` (see [ld.so - Linux
manual page](https://man7.org/linux/man-pages/man8/ld.so.8.html) for details),
so the correct Torch dependency should be found automatically; however, if you
encounter issues, you may have to modify the `LD_LIBRARY_PATH` environment
variable to also include the path to the Torch library (in addition to the
path to FTorch described above): 
```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path/to/Torch/installation>/lib
```

@Note
_Depending on your system and architecture `lib` may be `lib64` or something similar._
_On MacOS devices you will need to set `DYLD_LIBRARY_PATH` rather than `LD_LIBRARY_PATH`._
@endnote

@Note
If you wish to build your model/code with static linking it is possible to build FTorch 
as both a shared or a static library. For more information see the
[static vs. shared guidance](|page|/installation/general.html#building-ftorch-as-a-shared-vs-static-library)
on the main installation page.
@endnote

### Debug builds

Whilst experimenting, it may be useful to [build FTorch](|page|/installation/general.html)
using the `CMAKE_BUILD_TYPE=Debug` CMake flag (see
[CMAKE_BUILD_TYPE](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)
and [CMAKE_<LANG\>_FLAGS](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_FLAGS.html))
to allow useful error messages and investigation with debugging tools.


## Parallelism

If you are investigating running FTorch on HPC then you are probably interested
in improving computational efficiency via parallelism.

For a worked example of running with MPI, see the
[associated example](https://github.com/Cambridge-ICCS/FTorch/tree/main/example/7_MPI).

For information on running on GPU architectures, see the
[GPU user guide page](|page|/installation/gpu.html) and/or the
[MultiGPU example](https://github.com/Cambridge-ICCS/FTorch/tree/main/example/6_MultiGPU).
