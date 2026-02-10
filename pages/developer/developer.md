title: Developer Guide
author: Jack Atkinson
date: Last Updated: October 2025

## Developer Guide

This guide covers how to extend FTorch, contribute code, and the standards
expected for submissions.

- [Developer Requirements](#developer-requirements)
- [Extending the API](#extending-the-api)
    - [General guidelines](#general-guidelines)
    - [Fortran source generation using Fypp](#fortran-source-generation-using-fypp)
    - [GPU device handling](#gpu-device-handling)
- [Contribution Guidelines](#contribution-guidelines)
    - [Code style and standards](#code-style-and-standards)
- [Documentation](#documentation)
    - [In-code Documentation](#in-code-documentation)
    - [Written](#written-documentation)
    - [Versioning and Changelog](#versioning-and-changelog)


### Developer requirements

Development tools for [pre-processing](#fortran-source-and-fypp),
[code styling](#code-style) etc. are pip-installable using the
[requirements-dev file](https://github.com/Cambridge-ICCS/FTorch/blob/main/requirements-dev.txt):
```sh
pip install -r requirements-dev.txt
```

In order to streamline the process of uploading we provide a pre-commit hook in
[`.githooks/pre-commit`](https://github.com/Cambridge-ICCS/FTorch/blob/main/.githooks/pre-commit).
This will check that both the `.fypp` and `.f90` files have been updated together in a
synchronous fashion before a commmit can take place
([see below](#fortran-source-generation-using-fypp)).
Use of the hook is not automatic and needs to be enabled by the developer
(after they have inspected it and are happy with its contents).
Hooks can be enabled by placing them in the `.git` directory with the following commands:
```
cp .githooks/pre-commit .git/hooks/
chmod +x .git/hooks/pre-commit
```


### Extending the API

#### General guidelines

If you wish to add Torch functionality from the C++ API to
the FTorch Fortran API the steps are generally as follows:

* Modify `ctorch.cpp` to create a C++ version of the function that accesses `torch::<item>`.
* Add the function to the header file `ctorch.h`
* Modify `ftorch.fypp` to create a Fortran version of the function
  that binds to the version in `ctorch.cpp`.

Refer to the [LibTorch C++ Documentation](https://pytorch.org/cppdocs/)
and [C++ API documentation](https://pytorch.org/cppdocs/api/library_root.html)
for details of the available functions.

The following guidelines should be followed whilst writing new routines:

* Match optional argument defaults between Fortran, C, and C++
  ([principle of least astonishment](https://en.wikipedia.org/wiki/Principle_of_least_astonishment)).
* Handle `torch::Error` and `std::exception` in the C++ functions by catching and
  printing to screen before exiting cleanly.

#### Fortran source generation using Fypp

The Fortran source code in `src/ftorch_tensor.f90` should not be edited directly
but instead generated from `src/ftorch_tensor.fypp` by running the
[Fypp](https://fypp.readthedocs.io/en/stable/index.html) preprocessor.
This is done to simplify the process of overloading functions for multiple data
Fypp can be installed with the [developer requirements](#installing-developer-requirements).

To generate the Fortran code run:
```sh
fypp src/ftorch_tensor.fypp src/ftorch_tensor.f90
```

Conformance of these files is checked using GitHub continuous integration and
the [provided pre-commit hook](#developer-requirements).

@note
Generally it would be advisable to provide only the `.fypp` source code to
reduce duplication and confusion. However, because it is a relatively small file
and many of our users wish to _"clone-and-go"_ rather than develop, we provide both.<br>
Development should only take place in `ftorch_tensor.fypp`, however._
@endnote

The same applies to `ftorch_test_utils.fypp` and `ftorch_test_utils.f90`.

#### GPU device handling

GPU device-specific code is handled in FTorch using codes defined in the root
`CMakeLists.txt` file:
```cmake
set(GPU_DEVICE_NONE 0)
set(GPU_DEVICE_CUDA 1)
set(GPU_DEVICE_XPU 12)
set(GPU_DEVICE_MPS 13)
```
These are chosen to be consistent with the numbering used
[in PyTorch](https://github.com/pytorch/pytorch/blob/main/c10/core/DeviceType.h).

When a user specifies `-DGPU_DEVICE=XPU` (for example) in the FTorch CMake build, this
is mapped to the appropriate device code (in this case 12). Device codes
are passed to the C++ compiler in the following step:
```cmake
target_compile_definitions(
  ${LIB_NAME}
  PRIVATE GPU_DEVICE=${GPU_DEVICE_CODE}
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

@note
_The HIP/ROCm backend uses the same API as the CUDA backend, so FTorch treats
HIP as CUDA in places when calling LibTorch or PyTorch.
This should not concern end-users as the FTorch and pt2ts.py APIs handle this.
For further information see the
[PyTorch HIP documentation](https://docs.pytorch.org/docs/stable/notes/hip.html)_
@endnote


### Contribution Guidelines

Contributions for new features, bugfixes, or improvements should be raised in a pull
request.

In addition to making code contributions as [described above](#extending-the-api)
users should also include [documentation](#documentation), in-code and written if
required, and tests to cover any changes/additions.
Guidance on testing can be found on the [testing page](|page|/developer/testing.html).
Notable changes should also be documented in the [Changelog](#versioning-and-changelog).

#### Code style and standards

FTorch source code is subject to a number of static analysis checks to ensure that it
conforms to quality and legibility. These tools are a mixture of formatters and linters.

The tools we use are as follows on a language-by-language basis:

* Fortran: [fortitude](https://github.com/PlasmaFAIR/fortitude)
* C++: [clang-format](https://clang.llvm.org/docs/ClangFormat.html) and
  [clang-tidy](https://clang.llvm.org/extra/clang-tidy/)
* C: [clang-format](https://clang.llvm.org/docs/ClangFormat.html) and
  [clang-tidy](https://clang.llvm.org/extra/clang-tidy/)
* Python: [ruff](https://docs.astral.sh/ruff/)
* Shell: [ShellCheck](https://github.com/koalaman/shellcheck)
* CMake: cmake-lint from [cmake-format](https://github.com/cheshirekow/cmake_format)
  (Note: We do not use cmake-format's formatter)
* GitHub Actions workflows: [zizmor](https://woodruffw.github.io/zizmor)

Instructions on using these tools can be found in their respective documentations.
Note that all but ShellCheck may be installed with pip as described in the
[developer requirements](#developer-requirements).

Contributors should run these tools over their code and ensure that it conforms before
submitting a pull request.
If there is a good reason to ignore a particular rule this should be
justified in the pull request and ideally documented in the code.
There is a GitHub action as part of the continuous integration that will perform these
checks on all pull requests.


### Documentation

The documentation for FTorch is generated using
[FORD](https://forddocs.readthedocs.io/en/latest/) which is installed as part of the
[developer requirements](#developer-requirements).
This builds API documentation based off of in-code docstring syntax, and
web-based documentation from markdown pages.
For detailed information refer to the
[FORD User Guide](https://forddocs.readthedocs.io/en/latest/user_guide/index.html).

To generate the documentation run:
```
ford FTorch.md
```
from the root of the repository.

FORD uses [graphviz](https://graphviz.org) to generate dependency graphs from the Fortran
source code[^2].
For this, you will need to install it on your system - see the [installation guide](https://graphviz.org/download/) for your platform.

[^2]: Note: If FORD cannot locate the graphviz executable (it is not a hard dependency)
it will generate a warning.

#### In-code Documentation

Ford makes use of a docstring syntax for annotating code.
As a quick-start:

* `!!` is used to signify documentation.
* Documentation comes _after_ whatever it is documenting (inline or subsequent line).
* Documentation can precede an item if designated using `!>`.

The following examples from FORD and FTorch show this in context:
```fortran
subroutine feed_pets(cats, dogs, food)
    !! Feeds your cats and dogs, if enough food is available.

    ! Arguments
    integer, intent(in)  :: cats  !! The number of cats.
    integer, intent(in)  :: dogs  !! The number of dogs.
    real, intent(inout)  :: food
        !! The ammount of pet food (in kilograms) which you have on hand.

    !...

end subroutine feed_pets
```
```fortran
!> Type for holding a torch neural net (nn.Module).
type torch_model
    type(c_ptr) :: p = c_null_ptr  !! pointer to the neural net in memory
end type torch_model
```

Documentation of the C/C++ functions is provided
by [Doxygen](https://www.doxygen.nl/index.html).
This should be included in the header file `ctorch.h`.

#### Written Documentation

`FTorch.md` is the FORD index file that contains project metadata and describes
the project homepage.
Additional pages are contained in `pages/` as markdown files.

Notes:

- We need to define macros for GPU devices that are passed to FTorch via the `ftorch_types.F90` module.
  via the C preprocessor in `FTorch.md` to match those in the CMakeLists.txt.
- If building documentation locally you can set the `dbg: true` in `FTorch.md` to allow
  FORD to continue when encountering errors. Note that in this case the documentation
  may build but be incomplete.
- When writing new pages you can set `graph: false` in `FTorch.md` whilst prototyping
  to skip the time-consuming generation of dependency graphs.

#### Versioning and Changelog

FTorch has follows [semantic versioning](https://semver.org/).

- Major releases for API changes
- Minor releases periodically for new features
- Patches for bug fixes

The project version should be updated accordingly through the `PACKAGE_VERSION` in
CMakeLists.txt for each new release.

A log of notable changes to the software is kept in `CHANGELOG.md`.
This follows the conventions of [Keep a Changelog](https://keepachangelog.com/) and should
be updated by contributors and maintainers as part of a pull request when appropriate.

@note
"Notable" includes new features, bugfixes, dependency updates etc.

"Notable" does not cover typo corrections, documentation rephrasing and restyling,
or correction of other minor infelicities that do not impact the user or developer.
@endnote

New minor releases are made when deemed appropriate by maintainers by adding a tag to
the commit and creating a corresponding GitHub Release.
The minor number of the version should be incremented, the entry for the version
finalised in the changelog, and a clean log for 'Unreleased' changes created.

New patch releases are made whenever a bugfix is merged.
The patch number of the version should be incremented, a tag attached to the commit,
and a note made under the current 'Unreleased' patches header in the changelog.
