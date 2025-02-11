rem ---
rem Execute this shell script to run all of FTorch's integration tests.
rem
rem Assumes FTorch has been built with the `-DCMAKE_BUILD_TESTS=TRUE` option.
rem The `BUILD_DIR` variable in this script should be updated as appropriate for
rem your configuration.
rem
rem See `test/README.md` for more details on integration testing.
rem ---

rem NOTE: This version of run_test_suite only runs the integration tests, not
rem       the unit tests. These are not currently supported on Windows.

for /d %%i in (1_SimpleNet 2_ResNet18 4_MultiIO) do (
pushd build\examples\%%i
rem run the tests
ctest
rem The following line will propagate the error back to the cmd shell
rem This is necessary for the CI to detect a failed test
if %errorlevel% neq 0 exit /b %errorlevel%
popd
)
