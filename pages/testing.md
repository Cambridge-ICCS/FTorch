title: FTorch test suite

[TOC]

## Testing

FTorch's test suite is currently comprised of unit tests, as well as integration
tests based on a subset of the [examples](examples.html). These tests are built
and run and their outputs are analysed.

### Building

To enable FTorch's test suite, ensure that the `CMAKE_BUILD_TESTS` option
is set to `TRUE` for the build,  i.e., `-DCMAKE_BUILD_TESTS=True`. If you want
to run the unit tests, you will also need to install
[pFUnit](https://github.com/Goddard-Fortran-Ecosystem/pFUnit) and pass its
install directory to the `CMAKE_PREFIX_PATH` when building FTorch.

Note that, whilst example `5_Looping` is built if `CMAKE_BUILD_TESTS=TRUE` is
specified, it is not run as part of the integration test suite because it
demonstrates 'good' versus 'bad' practice, as opposed to functionality.

### Running

Once the build is complete, activate the Python virtual environment you created
for FTorch<sup>1</sup> and simply run the
[helper script](https://github.com/Cambridge-ICCS/FTorch/blob/main/run_test_suite.sh)
in the root FTorch directory. Depending on the OS you are running you will need
to use either:

- `./run_test_suite.sh` for Unix operating systems (Mac and Linux).
- `run_test_suite.bat` for Windows operating systems (note this will only run
  the integration tests, not unit tests).

This will automatically install any additional Python dependencies for the
examples.

See the subsections below for instructions on how to run subsets of the full
test suite.

> <sup>1</sup> _If you built FTorch against LibTorch (rather than creating a
virtual environment) then you will need to
[create a virtual environment](https://docs.python.org/3/library/venv.html) for
the purposes of integration testing as this script will install packages into your
Python environment and will check that a virtual environment is in use._

#### Running unit tests on Unix

If you are running with a Unix operating system, unit tests may be
executed either as a suite by specifying the `--unit-only` command line
argument:
```sh
./run_test_suite.sh --unit-only
```
or individually by navigating to `${BUILD_DIR}/test/unit` (where `${BUILD_DIR}`
is the build directory for FTorch) and calling `ctest` with
[appropriate arguments](https://cmake.org/cmake/help/latest/manual/ctest.1.html).
This will produce a report on which tests passed and which failed for your
build.

#### Running integration tests on Unix

If you are running with a Unix operating system, integration tests may be
executed either as a suite by specifying the `--integration-only` command line
argument:
```sh
./run_test_suite.sh --integration-only
```
or individually by navigating to the corresponding subdirectory of
`${BUILD_DIR}/examples` (where `${BUILD_DIR}` is the build directory for
FTorch) and calling `ctest`. This will produce a report on which tests passed
and which failed for your build. Note that some of the examples have additional
dependencies, which may need installing into your virtual environment.

#### Running integration tests on Windows

As mentioned above, if you are running with a Windows operating system then
only the integration tests are currently set up. They may be executed either as
a suite with the command
```sh
./run_test_suite.bat
```
or individually by navigating to the corresponding subdirectory of
`${BUILD_DIR}/examples` (where `${BUILD_DIR}` is the build directory for
FTorch) and calling `ctest`. This will produce a report on which tests passed
and which failed for your build. Note that some of the examples have additional
dependencies, which may need installing into your virtual environment.
