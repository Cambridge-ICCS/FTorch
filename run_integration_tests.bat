for /d %%i in (1_SimpleNet 2_ResNet18 4_MultiIO) do (
pushd build\test\examples\%%i
rem run the tests
ctest
rem The following line will propagate the error back to the cmd shell
rem This is necessary for the CI to detect a failed test
if %errorlevel% neq 0 exit /b %errorlevel%
popd
)
