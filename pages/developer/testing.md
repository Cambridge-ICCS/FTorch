title: FTorch test suite
author: Joe Wallwork
date: Last Updated: October 2025


## Testing

FTorch's test suite includes unit tests of components, and integration
tests based on a subset of the [worked examples](|page|/usage/worked_examples.html).

- [Building Tests](#building)
- [Running Tests](#running)
    - [Unit Tests](#unit-tests)
    - [Integration Tests](#integration-tests)
- [Contributing Tests](#contributing-tests)
    - [Contributing Unit Tests](#contributing-unit-tests)
    - [Contributing Integration Tests](#contributing-integration-tests)


### Building

To enable FTorch's test suite, set `CMAKE_BUILD_TESTS=TRUE` during the build.
To run the unit tests, you will need to install
[pFUnit](https://github.com/Goddard-Fortran-Ecosystem/pFUnit) and provide its
install location to the `CMAKE_PREFIX_PATH` or as the environment variable `PFUNIT_DIR`:
```shell
# Update `CMAKE_PREFIX_PATH` explicitly with pFUnit install directory
cmake -DCMAKE_PREFIX_PATH="</path/to/pFUnit/build/installed/PFUNIT-VERSION>" \
    -DCMAKE_BUILD_TESTS=True -S </path/to/FTorch> -B </path/to/FTorch/build>
```
or
```shell
# Using an environment variable with pFUnit install directory
export PFUNIT_DIR=</path/to/pFUnit/build/installed/PFUNIT-VERSION>
cmake -DCMAKE_BUILD_TESTS=TRUE -S </path/to/FTorch> -B </path/to/FTorch/build>
```

Note that pFUnit includes the version number in the install directory name,
so for version 4.12 that path will need to be specified as
`/path/to/pFUnit/build/installed/PFUNIT-4.12`, for example.

@note
If a `GPU_DEVICE` is specified but only one is available, set `MULTI_GPU=OFF` to skip
the 'multiple GPU devices' integration test.
@endnote

Building with tests enabled will automatically install any Python
dependencies for the examples, so should be executed from within a virtual environment.[^1] If this is not the case it will fail with appropriate warnings.

[^1]: _If you built FTorch against LibTorch (rather than creating a
virtual environment) then you will need to
[create a virtual environment](https://docs.python.org/3/library/venv.html) for
the purposes of integration testing as this script will install packages into your
Python environment and will check that a virtual environment is in use._

Note that, whilst example `5_Looping` is built if `CMAKE_BUILD_TESTS=TRUE` is
specified, it is not run as part of the integration test suite because it
demonstrates 'good' versus 'bad' practice, as opposed to functionality.


### Running

Ensure that the Python virtual environment used when
building is active and then run `ctest` from the build directory to execute all tests.
Use [ctest arguments](https://cmake.org/cmake/help/latest/manual/ctest.1.html)
for greater control over the testing configuration.

This will produce a report on which of the requested tests passed, and which, if any,
failed for your build.

#### Unit tests

Unit tests may be executed in the following ways:

1. To run just the unit tests (tests whose names start with 'unittest') use
   `ctest -R unittest`.
2. To run a specific unit test use
   `ctest -R unittest_tensor_constructors_destructors`, for example.

#### Integration tests

Integration tests may be executed in the following ways:

1. To run just the integration tests (tests whose names start with 'example') use
   `ctest -R example`
2. To run a specific integration test use `ctest -R example2`, for example.<br>
   Alternatively navigate to the corresponding example in `${BUILD_DIR}/examples`
   and call `ctest`.


### Contributing tests

#### Contributing unit tests

New components should come with unit tests written using the
[pFUnit](https://github.com/Goddard-Fortran-Ecosystem/pFUnit) framework.

- New unit tests should be added to the `test/unit/` directory and start with
  `unittest_`.
- New unit tests should be included in `test/unit/CMakeLists.txt` in order to
  be built as part of the test suite.

#### Contributing integration tests

New functionalities should come with integration tests in the form of worked
examples.

- These should take the form of a new example in the `examples/` directory.
- Create a subdirectory named with the next sequential number and a descriptive
  name, e.g. `9_NewFeature`.
- In addition to a `CMakeLists.txt` to build the example code there
  should also be a section at the end setting up running of the example
  using [CTest](https://cmake.org/cmake/help/book/mastering-cmake/chapter/Testing%20With%20CMake%20and%20CTest.html).
    - Integration test names should start with `example_`
    - New examples will also need including in `examples/CMakeLists.txt`
- Ensure the documentation on [worked examples](|page|/usage/worked_examples.html) is
  updated accordingly.
