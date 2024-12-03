pushd %1
ctest
if %errorlevel% neq 0 exit /b %errorlevel%
popd
