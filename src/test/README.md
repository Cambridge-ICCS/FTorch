# Integration tests

This subdirectory is populated with code and data from the examples directory
during the test suite. The examples can be run as integration tests to verify
that FTorch is working as expected in those cases.

Currently, only the first ('SimpleNet') and second ('ResNet18') examples have
been implemented as integration tests. To build them as tests, add the flag
```
-DCMAKE_BUILD_TESTS=TRUE
```
when building FTorch. To run them, go to the corresponding subdirectory of
`src/build/test/examples` and call `ctest`, which will produce a report on
which tests passed and which failed for your build.
