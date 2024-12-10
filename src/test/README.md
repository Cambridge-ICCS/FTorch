# Integration tests

This subdirectory is populated with code and data from the examples directory
during the test suite. The examples can be run as integration tests to verify
that FTorch is working as expected in those cases.

Currently, only the first ('SimpleNet') and second ('ResNet18') examples have
been implemented as integration tests.

## Building

To build them as tests, add the flag
```
-DCMAKE_BUILD_TESTS=TRUE
```
when building FTorch.

## Running

Having built the integration tests, ensure the Python virtual environment is
active and run them by going to the corresponding subdirectory of
`src/build/test/examples` and calling `ctest`. This will produce a report on
which tests passed and which failed for your build. Note that the examples have
additional dependencies, which may need installing into your virtual
environment.

Alternatively, run the helper script in the root FTorch directory. Depending on
which operating system you are running, you will need:

- `./run_integration_tests.sh` for unix (mac and linux)
- `run_integration_tests.bat` for windows

This will automatically install any additional Python dependencies for the
examples and run the tests.
