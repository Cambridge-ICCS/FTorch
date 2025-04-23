# Changelog

All [notable](https://cambridge-iccs.github.io/FTorch/page/developer.html#versioning-and-changelog)
changes to the project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
For specific details see the [FTorch online documentation](https://cambridge-iccs.github.io/FTorch/page/developer.html#versioning-and-changelog).

## [Unreleased](https://github.com/Cambridge-ICCS/FTorch/compare/v1.0.0...HEAD)

### Added

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

- fortitude dependency version increased to 0.7.0
- Examples reordered to be more logical in [#317](https://github.com/Cambridge-ICCS/FTorch/pull/317)
- scalar multiplication/division of tensors reworked to require the scalar to first be mapped to a `torch_tensor` in [#289](https://github.com/Cambridge-ICCS/FTorch/pull/289)
- The unit tests for constructing and destroying tensors were separated out in
  [#319](https://github.com/Cambridge-ICCS/FTorch/pull/319)
- Demo numbers were bumped to account for new demo in
  [#291](https://github.com/Cambridge-ICCS/FTorch/pull/291).
- Use interface for `torch_tensor_from_array` with default layout in tests and
  examples in [#348](https://github.com/Cambridge-ICCS/FTorch/pull/348).
- Error handling in `ctorch.cpp` improved in [#347](https://github.com/Cambridge-ICCS/FTorch/pull/347).

### Removed

- Windows CI disabled until GitHub runner issues resolved in [50ea6d7](https://github.com/Cambridge-ICCS/FTorch/commit/50ea6d78d79ebe638ebe597e745c015549f12a61)

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
