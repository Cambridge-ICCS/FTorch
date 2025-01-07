title: FTorch test suite

[TOC]

## Testing

FTorch's test suite is currently comprised of unit tests, as well as integration
tests based on a subset of the [examples](examples.html). These tests are built
and run and their outputs are analysed.

### Building

To enable FTorch's test suite, ensure that the `CMAKE_BUILD_TESTS` option
is set to `TRUE` for the build,  i.e., `-DCMAKE_BUILD_TESTS=True`.. For the unit
tests, you will also need to install
[pFUnit](https://github.com/Goddard-Fortran-Ecosystem/pFUnit) and pass its
install directory to the `CMAKE_PREFIX_PATH` when building FTorch.

### Running

Once the build is complete, activate the Python virtual environment you created
for FTorch<sup>1</sup> and simply run the
[helper script](https://github.com/Cambridge-ICCS/FTorch/blob/main/run_test_suite.sh)
in the root FTorch directory. Depending on the OS you are running you will need
to use either:
```
- `./run_test_suite.sh` for Unix (Mac and Linux)
- `run_test_suite.bat` for Windows (note this will only run the integration tests, not unit tests).

This will automatically install any additional Python dependencies for the
examples.

Alternatively, individual tests may be run by going to `${BUILD_DIR}/test/unit`
or the corresponding subdirectory of `${BUILD_DIR}/test/examples` (where
`${BUILD_DIR}` is the build directory for FTorch) and calling `ctest`. This
will produce a report on which tests passed and which failed for your build.
Note that some of the examples have additional dependencies, which may need
installing into your virtual environment.

> <sup>1</sup> _If you built FTorch against libtorch (rather than creating a
virtual environment) then either
[create a virtual environment](https://docs.python.org/3/library/venv.html) for
the purposes of testing, or note that this script may have your Python
environment install some modules._
