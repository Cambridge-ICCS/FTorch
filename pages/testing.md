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

The CMake build will automatically install any additional Python dependencies
for the examples.

Note that, whilst example `5_Looping` is built if `CMAKE_BUILD_TESTS=TRUE` is
specified, it is not run as part of the integration test suite because it
demonstrates 'good' versus 'bad' practice, as opposed to functionality.

### Running

Once the build is complete, activate the Python virtual environment you created
for FTorch<sup>1</sup> and simply call `ctest` from the build directory. Doing
this, you can pass
[arguments](https://cmake.org/cmake/help/latest/manual/ctest.1.html) to `ctest`
for greater control over the testing configuration.

See the subsections below for instructions on how to run subsets of the full
test suite.

> <sup>1</sup> _If you built FTorch against LibTorch (rather than creating a
virtual environment) then you will need to
[create a virtual environment](https://docs.python.org/3/library/venv.html) for
the purposes of integration testing as this script will install packages into your
Python environment and will check that a virtual environment is in use._

#### Running unit tests

Unit tests may be executed in the following ways:

1. Navigate to the build directory and call `ctest -R unittest` to run all unit
   tests. (This will run all tests whose names start with 'unittest'.)
2. Navigate to the build directory and call
   `ctest -R unittest_tensor_constructors_destructors` (for example) to run a
   specific unit test.

Either approach will produce a report on which of the requested tests passed
and which failed for your build.

#### Running integration tests

Integration tests may be executed in the following ways:

1. Navigate to the build directory and call `ctest -R example` to run all
   integration tests. (This will run all tests whose names start with
   'example').
2. Run the tests associated with a specific example by navigating to the build
   directory and calling `ctest -R example2`, for example.
3. Run the tests associated with a specific example by navigating to the
   corresponding subdirectory of `${BUILD_DIR}/examples` (where `${BUILD_DIR}`
   is the build directory for FTorch) and calling `ctest`.

Any of the above will produce a report on which of the requested tests passed
and which failed for your build.

Note that some of the examples have additional dependencies. While these will be
installed into your virtual environment as part of the CMake build, you should
ensure that the same virtual environment is active when you run the tests.
