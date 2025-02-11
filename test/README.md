# FTorch test suite

## Unit tests

The `unit` subdirectory contains unit tests written using
[pFUnit](https://github.com/Goddard-Fortran-Ecosystem/pFUnit). To be able to
compile these, you will need to first install pFUnit and then pass its install
directory to the `CMAKE_PREFIX_PATH` when building FTorch.

Note that the unit tests are currently only available on Unix based systems, not Windows.


## Integration tests

The `examples` subdirectory is populated with code and data from the examples
directory during the test suite. The examples can be run as integration tests to
verify that FTorch is working as expected in those cases.

Currently, only a subset of the examples have been implemented as integration
tests.

## Building

To build with testing enabled, add the flag
```
-DCMAKE_BUILD_TESTS=TRUE
```
when building FTorch.

## Running

Having built the unit and integration tests, ensure the Python virtual
environment is active and run them by going to `build/test/unit` or the
corresponding subdirectory of `build/test/examples` and calling `ctest`.
This will produce a report on which tests passed and which failed for your
build. Note that the examples have additional dependencies, which may need
installing into your virtual environment.

Alternatively, run the helper script in the root FTorch directory. Depending on
which operating system you are running, you will need:

- `./run_test_suite.sh` for unix (Mac and Linux)
- `run_test_suite.bat` for Windows (note this will only run the integration tests, not unit tests).

This will automatically install any additional Python dependencies for the
examples and run the tests.
