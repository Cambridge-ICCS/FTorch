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
environment is active and run them by navigating to the build directory and
calling `ctest`. For finer-grained control over which tests to run, see the
[testing user guide page](https://cambridge-iccs.github.io/FTorch/page/testing.html).
