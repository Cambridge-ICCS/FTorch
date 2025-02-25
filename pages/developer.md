title: Developer Guide

If you would like to contribute to the FTorch project, or modify the code at a deeper
level, please see below for guidance.

[TOC]


## Getting involved

Contributions and collaborations are welcome.

For bugs, feature requests, and clear suggestions for improvement please
[open an issue](https://github.com/Cambridge-ICCS/FTorch/issues/new/choose).

If you have built something upon _FTorch_ that would be useful to others, or can
address an [open issue](https://github.com/Cambridge-ICCS/FTorch/issues), please
[fork the repository](https://github.com/Cambridge-ICCS/FTorch/fork) and open a
pull request.


### Code of Conduct
Everyone participating in the FTorch project, and in particular in the
issue tracker, pull requests, and social media activity, is expected to treat other
people with respect and, more generally, to follow the guidelines articulated in the
[Python Community Code of Conduct](https://www.python.org/psf/codeofconduct/).


## Extending the API

If you have a Torch functionality that you wish to bring in from the C++ API to
the FTorch Fortran API the steps are generally as follows:

* Modify `ctorch.cpp` to create a C++ version of the function that accesses `torch::<item>`.
* Add the function to the header file `ctorch.h`
* Modify `ftorch.fypp` to create a Fortran version of the function
  that binds to the version in `ctorch.cpp`.

Details of C++ functionalities available to be wrapped can be found
in the [LibTorch C++ API](https://pytorch.org/cppdocs/).

As this is an open-source project we appreciate any contributions
back from users that have extended the functionality.
If you have done something but don't know where to start with
open-source contributions please get in touch!<sup>*</sup>

<sup>*</sup>_Our preferred method of contact is via Github issues and discussions,
but if you are unfamiliar with this you can [email ICCS](mailto:jwa34@cam.ac.uk)
asking for the FTorch developers._


### Fortran source and Fypp

The Fortran source code for FTorch is contained in `src/ftorch.f90`.
However, this file should not be edited directly, but instead generated from
`src/ftorch.fypp`.
This is a file that is set up to be run through the
[Fypp](https://fypp.readthedocs.io/en/stable/index.html) preprocessor.
We use this because we want to create a pleasant interface of single function calls.
The nature of Fortran means that this requires a lot of repeated combinations of
array shapes and data types under interface structures.
By using Fypp we can generate these programatically.

Fypp can be installed via pip:
```
pip install fypp
```

To generate the Fortran code run:
```
fypp src/ftorch.fypp src/ftorch.f90
```

_Note: Generally it would be advisable to provide only the `.fypp` source code to
reduce duplication and confusion. However, because it is a relatively small file
and many of our users wish to _"clone-and-go"_ rather than develop, we provide both.<br>
Development should only take place in `ftorch.fypp`, however._


### Torch C++ API

When extending or modifying functionality related to C++ header and/or source
files `src/ctorch.h` and `src/ctorch.cpp`, we refer to the Torch
[C++ documentation](https://pytorch.org/cppdocs) and more specifically the
[C++ API documentation](https://pytorch.org/cppdocs/api/library_root.html)
pages on the PyTorch website for details.

### GPU device handling

GPU device-specific code is handled in FTorch using codes defined in the root
`CMakeLists.txt` file:
```cmake
set(GPU_DEVICE_NONE 0)
set(GPU_DEVICE_CUDA 1)
set(GPU_DEVICE_XPU 12)
set(GPU_DEVICE_MPS 13)
```
These device codes are chosen to be consistent with the numbering used in
PyTorch (see
https://github.com/pytorch/pytorch/blob/main/c10/core/DeviceType.h). When a user
specifies `-DGPU_DEVICE=XPU` (for example) in the FTorch CMake build, this is
mapped to the appropriate device code (in this case 12). The chosen device code
and all other ones defined are passed to the C++ compiler in the following step:
```cmake
target_compile_definitions(
  ${LIB_NAME}
  PRIVATE ${COMPILE_DEFS} GPU_DEVICE=${GPU_DEVICE_CODE}
          GPU_DEVICE_NONE=${GPU_DEVICE_NONE} GPU_DEVICE_CUDA=${GPU_DEVICE_CUDA}
          GPU_DEVICE_XPU=${GPU_DEVICE_XPU} GPU_DEVICE_MPS=${GPU_DEVICE_MPS})
```
The chosen device code will enable the appropriate C pre-processor conditions in
the C++ source so that that the code relevant to that device type becomes
active.

An example illustrating why this approach was taken is that if we removed the
device codes and pre-processor conditions and tried to build with a CPU-only or
CUDA LibTorch installation then compile errors would arise from the use of the
`torch::xpu` module in `src/ctorch.cpp`.

### git hook

In order to streamline the process of uploading we provide a pre-commit hook in
[`.githooks/pre-commit`](https://github.com/Cambridge-ICCS/FTorch/blob/main/.githooks/pre-commit).
This will check that both the `.fypp` and `.f90` files have been updated together in a
synchronous fashion before a commmit can take place.
If this does not happen then the second line of defence (GitHub continuous integration)
will fail following the commit.

Use of the hook is not automatic and needs to be enabled by the developer
(after they have inspected it and are happy with its contents).
Hooks can be enabled by placing them in the `.git` directory with the following commands:
```
cp .githooks/pre-commit .git/hooks/
chmod +x .git/pre-commit
```


### Code style

FTorch source code is subject to a number of static analysis checks to ensure that it
conforms to quality and legibility. These tools are a mixture of formatters and linters.

The tools we use are as follows on a language-by-language basis:

* Fortran: [fortitude](https://github.com/PlasmaFAIR/fortitude)
* C++: [clang-format](https://clang.llvm.org/docs/ClangFormat.html) and [clang-tidy](https://clang.llvm.org/extra/clang-tidy/)
* C: [clang-format](https://clang.llvm.org/docs/ClangFormat.html) and [clang-tidy](https://clang.llvm.org/extra/clang-tidy/)
* Python: [ruff](https://docs.astral.sh/ruff/)
* Shell: [ShellCheck](https://github.com/koalaman/shellcheck)
* CMake: [cmake-format](https://github.com/cheshirekow/cmake_format)
* GitHub Actions workflows: [zizmor](https://woodruffw.github.io/zizmor)

Instructions on installing these tools can be found in their respective documentations.
Note that all but ShellCheck may be installed with pip. A shortcut for doing
this is to run the following from the base FTorch directory:
```
pip install -r requirements.txt
```

Contributors should run them over their code and ensure that it conforms before submitting
a pull request. If there is a good reason to ignore a particular rule this should be
justified in the pull request and ideally documented in the code.

There is a GitHub action as part of the continuous integration that will perform these
checks on all opened pull requests before they are merged.


### General guidelines

* Match optional argument defaults between Fortran, C, and C++<br>
  ([principle of least astonishment](https://en.wikipedia.org/wiki/Principle_of_least_astonishment)).
* Handle `torch::Error` and `std::exception` in the C++ functions by catching and
  printing to screen before exiting cleanly.


## Documentation

The API documentation for FTorch is generated using 
[FORD](https://forddocs.readthedocs.io/en/latest/).
For detailed information refer to the 
[FORD User Guide](https://forddocs.readthedocs.io/en/latest/user_guide/index.html),
but as a quick-start:

* `!!` is used to signify documentation.
* Documentation comes _after_ whatever it is documenting (inline or subsequent line).
* Documentation can precede an item if designated using `!>`.

FORD is pip installable:
```
pip install ford
```
To generate the documentation run:
```
ford FTorch.md
```
from the root of the repository.

`FTorch.md` is the FORD index file, API documentation is automatically generated, and
any further items are contained in `pages/` as markdown files.

Documentation of the C functions in `ctorch.h` is provided
by [Doxygen](https://www.doxygen.nl/index.html).

Note that we need to define the macros for GPU devices that are passed to `ftorch.F90`
via the C preprocessor in `FTorch.md` to match those in the CMakeLists.txt.
