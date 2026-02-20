# Changelog

All [notable](https://cambridge-iccs.github.io/FTorch/page/developer/developer.html#versioning-and-changelog)
changes to the project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
For specific details see the [FTorch online documentation](https://cambridge-iccs.github.io/FTorch/page/developer.html#versioning-and-changelog).

## [Unreleased](https://github.com/Cambridge-ICCS/FTorch/compare/v1.0.0...HEAD)

### Added

- Add documentation on FTorch's unit testing approach. [#489](https://github.com/Cambridge-ICCS/FTorch/pull/489)
- Batching in FTorch documented, including a new worked example, in
  [#500](https://github.com/Cambridge-ICCS/FTorch/pull/500).
- Add subroutine for printing parameters associated with a Torch Model
  [#488](https://github.com/Cambridge-ICCS/FTorch/pull/488)
- Add unit testing for the Torch Model API [#496](https://github.com/Cambridge-ICCS/FTorch/pull/496)
- Provide support for using FTorch with pkg-config. [#464](https://github.com/Cambridge-ICCS/FTorch/pull/464)
- Support building FTorch as a static library. [#448](https://github.com/Cambridge-ICCS/FTorch/pull/448)
- Intel-ifx and Intel-ifort CI and GCC v9-13 CI. Intel CI builds OpenMPI from source to accomodate MPI integration tests [#438](https://github.com/Cambridge-ICCS/FTorch/pull/438)
- Expose tensor strides via `get_stride` method [#416](https://github.com/Cambridge-ICCS/FTorch/pull/416)
- Remove `UNIX` preprocessor variable that selected the  right C-integer type for 64bit int. Use `int64_t` instead [#416](https://github.com/Cambridge-ICCS/FTorch/pull/416)
- A new cmake option `MULTI_GPU` to control the build of multi GPU integration tests in [#410](https://github.com/Cambridge-ICCS/FTorch/pull/410)
- Support for AMD GPU backends (HIP) provided in
  [#385](https://github.com/Cambridge-ICCS/FTorch/pull/385) and
  [#388](https://github.com/Cambridge-ICCS/FTorch/pull/388).
- `requires_grad` property hooked up to `torch_tensor` in [#288](https://github.com/Cambridge-ICCS/FTorch/pull/288)
- MPI example added in [#270](https://github.com/Cambridge-ICCS/FTorch/pull/270)
- Changelog file and guidance for versioning added in [#313](https://github.com/Cambridge-ICCS/FTorch/pull/313)
- A new tensor manipulation demo was introduced in [#291](https://github.com/Cambridge-ICCS/FTorch/pull/291).
- Backpropagation implemented with `torch_tensor_backward` and
  `torch_tensor_get_gradient` in [#286](https://github.com/Cambridge-ICCS/FTorch/pull/286)
- Zeroing of gradients associated with a tensor implemented in
  [#341](https://github.com/Cambridge-ICCS/FTorch/pull/341).
- Exposed `retain_graph` argument for `torch_tensor_backward` in
  [#342](https://github.com/Cambridge-ICCS/FTorch/pull/342).
- Implemented `torch_tensor_zero` and class method alias in
  [#338](https://github.com/Cambridge-ICCS/FTorch/pull/338).
- Provided interface for `torch_tensor_from_array` with default layout in
  [#348](https://github.com/Cambridge-ICCS/FTorch/pull/348).
- Overload taking sum and mean of tensors in
  [#344](https://github.com/Cambridge-ICCS/FTorch/pull/344).

### Changed

- Improve FTorch's modularisation under the hood by breaking single Fortran file down into submodules in [#508](https://github.com/Cambridge-ICCS/FTorch/pull/508)
- Example numbers were bumped to account for new worked examples in
  [#291](https://github.com/Cambridge-ICCS/FTorch/pull/291) and
  [#500](https://github.com/Cambridge-ICCS/FTorch/pull/500).
- Bump the minimum CMake version from 3.15 to 3.18, for consistency with what's
  used in PyTorch. [#491](https://github.com/Cambridge-ICCS/FTorch/pull/491)
- Significant overhaul of the online FORD documentation and reduction of content in the README
  in [#459](https://github.com/Cambridge-ICCS/FTorch/pull/459)
- Intel CI now uses Intel oneAPI MPI instead of OpenMPI built with Intel compilers [#449](https://github.com/Cambridge-ICCS/FTorch/pull/449)
- FTorch library (`libftorch.so`) produced by cmake installation now has `RUNPATH` that contains path to Torch library directory. Downstream targets linking against FTorch can now find the Torch dependency automatically and will compile successfully [#437](https://github.com/Cambridge-ICCS/FTorch/pull/437).
- In all `CMakeLists.txt` where `find_package(FTorch)` was present, now using `REQUIRE` if not building tests to stop the cmake configuation process early for users who only wish to build examples in [#434](https://github.com/Cambridge-ICCS/FTorch/pull/434)
- fortitude dependency version increased to 0.7.0
- Examples reordered to be more logical in [#317](https://github.com/Cambridge-ICCS/FTorch/pull/317)
- scalar multiplication/division of tensors reworked to require the scalar to first be mapped to a `torch_tensor` in [#289](https://github.com/Cambridge-ICCS/FTorch/pull/289)
- The unit tests for constructing and destroying tensors were separated out in
  [#319](https://github.com/Cambridge-ICCS/FTorch/pull/319)
- Use interface for `torch_tensor_from_array` with default layout in tests and
  examples in [#348](https://github.com/Cambridge-ICCS/FTorch/pull/348).
- Error handling in `ctorch.cpp` improved in [#347](https://github.com/Cambridge-ICCS/FTorch/pull/347).

### Removed

- Windows CI disabled until GitHub runner issues resolved in [50ea6d7](https://github.com/Cambridge-ICCS/FTorch/commit/50ea6d78d79ebe638ebe597e745c015549f12a61)

### Fixed

- Make input array for `torch_tensor_from_array` have the `pointer, contiguous`
  properties rather than `target`
  [#530](https://github.com/Cambridge-ICCS/FTorch/pull/530). This change
  technically breaks the API because it becomes no longer possible to pass
  temporary Fortran arrays to the second argument of `torch_tensor_from_array`.
  However, that was a bug rather than a feature, so any workflow crashes due to
  this change will provide the user with information on how to remove the error.

### Patch Releases


## [1.0.0](https://github.com/Cambridge-ICCS/FTorch/releases/tag/v1.0.0) - 2025-03-05

### Added

- First release of FTorch accompanying pulication in JOSS
- MIT License
- Notable features of the library include:
  - Representation of Torch tensors and models in Fortran
  - Ability to run inference of Torch models from Fortran
  - Early implementation of autograd features for Torch tensors in Fortran
- Comprehensive examples suite showcasing usage
- Testing suites:
  - Unit, using [pFUnit](https://github.com/Goddard-Fortran-Ecosystem/pFUnit)
  - Integration, based on examples
- Code quality and static analysis checks
- Documentation:
  - README.md and associated files in repository
  - Online API and comprehensive docs build using [FORD](https://forddocs.readthedocs.io/)
