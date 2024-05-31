title: FTorch test suite

[TOC]

## Testing

FTorch's test suite is currently comprised of integration tests based on a
subset of the [examples](examples.html). These tests are built and run and their
outputs are analysed to check they contain expected regular expressions.

### Building the integration tests

To enable FTorch's integration tests, ensure that the `CMAKE_BUILD_TESTS` option
is set to `TRUE` for the build.

### Running the integration tests

Once the build is complete, activate the Python virtual environment you created
for FTorch and run an individual test by going to the desired subdirectory of
`${BUILD_DIR}/test/examples` (where `${BUILD_DIR}` is the build directory for
FTorch) and calling `ctest`. This will produce a report on which tests passed
and which failed for your build. Note that some of the examples have additional
dependencies, which may need installing into your virtual environment.

As an alternative to the above, simply run the
[helper script](https://github.com/Cambridge-ICCS/FTorch/blob/main/run_integration_tests.sh)
in the root FTorch directory:
```
./run_integration_tests.sh
```
This will automatically install any additional dependencies for the examples.
