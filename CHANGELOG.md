# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added


### Changed

- fortitude dependency version increased to 0.7.0

### Removed


## [1.0.0] - 2025-03-05

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

[unreleased]: https://github.com/Cambridge-ICCS/FTorch/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Cambridge-ICCS/FTorch/releases/tag/v1.0.0
