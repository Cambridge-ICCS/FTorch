title: System-Specific Guidance
author: Jack Atkinson
date: Last Updated: October 2025

## System-Specific Guidance

- [Windows](#windows)
- [Apple Silicon](#apple-silicon)
- [Conda](#conda)
- [GitHub Codespace](#github-codespace)

### Windows

If possible we recommend using the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/) (WSL) to build
the library. In this case the build process is the same as for a Linux environment.

Building in Windows itself can be done using Visual Studio and the Intel Fortran
resources. The following additional dependencies are also required:

* [Visual Studio](https://visualstudio.microsoft.com/) ensuring C++ tools are selected and installed.
* [Intel OneAPI Basetoolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
* [Intel OneAPI HPC toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit.html) ensuring that the Intel Fortran compiler and VS integration is selected.

Note that LibTorch is _not_ supported for the GNU Fortran compiler with MinGW.

#### Installation

Load the Intel Fortran compilers using `setvars.bat` which is found in the Intel compiler install
directory (see the [Intel
docs](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2023-2/use-the-setvars-script-with-windows.html))
for more details.

From `cmd` this can be done with:
```
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```

FTorch can then be built according to the [regular CMake instructions](|page|/installation/general.html),
with the addition of `-G "NMake Makefiles"`.

So the basic command to build from `cmd` becomes:
```
cmake -G "NMake Makefiles" -DCMAKE_PREFIX_PATH="C:\Users\<path-to-libtorch-download>\libtorch" -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
cmake --install .
```

> Note: _In a Windows environment administrator privileges are required for the default install location._


The following is an example `cmd` script that installs FTorch and runs the integration tests. It assumes you have already
installed CMake, git, the Intel compilers, and Visual Studio. There are a few
places where output is turned on or off using the `ECHO` command. If you are
experiencing issues with the install then it may be helpful to set `ECHO ON`
throughout.

```cmd
rem Disable output for now
ECHO OFF

rem Load intel compilers
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

rem Download ftorch
git clone https://github.com/Cambridge-ICCS/FTorch.git
cd FTorch

rem Make virtual environment
python -m venv .ftorch

rem Activate the virtual environment
call .ftorch\Scripts\activate

rem Install torch
pip install torch torchvision torchaudio

rem Enable output
ECHO ON

rem Find Torch location
for /f "tokens=2*" %%i in ('pip show torch ^| findstr /R "^Location"') do set torch_path=%%i

rem Run CMake to generate build scripts
rem (Update CMAKE_PREFIX_PATH depending on location of ftorch venv)
cmake -Bbuild -G "NMake Makefiles" -DCMAKE_Fortran_FLAGS="/fpscomp:logicals" ^
 -DCMAKE_PREFIX_PATH="%torch_path%" ^
 -DCMAKE_BUILD_TYPE=Release ^
 -DCMAKE_BUILD_TESTS=True ^
 -DCMAKE_Fortran_COMPILER=ifx ^
 -DCMAKE_C_COMPILER=icx ^
 -DCMAKE_CXX_COMPILER=icx
 -DCMAKE_Fortran_FLAGS="/fpscomp:logicals" ^
 -DCMAKE_CXX_FLAGS="/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"

rem Build and install FTorch
cmake --build build
cmake --install build

rem Quit if this raises an error
if %errorlevel% neq 0 exit /b %errorlevel%

ECHO OFF
rem Add FTorch and PyTorch libs to path
rem (Update the first one depending on where you installed FTorch)
set PATH=C:\Program Files (x86)\FTorch\bin;%PATH%
set PATH=%torch_path%;%PATH%
set PATH=%torch_path%\torch\lib;%PATH%

rem Run integration tests
ECHO ON
ctest --verbose --tests-regex example1
ctest --verbose --tests-regex example2
ctest --verbose --tests-regex example3
ctest --verbose --tests-regex example4
ctest --verbose --tests-regex example8
if %errorlevel% neq 0 exit /b %errorlevel%
```

Here the `/fpscomp:logicals` flag is used to ensure Fortran logicals are
compatible with those used by PyTorch. The
`/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` flag is used to suppress warnings
related to mismatched compiler versions between the Intel compilers and
those used to build LibTorch.

We recommend Windows users review the Windows continuous integration workflow
([`.github/workflows/test_suite_windows_cpu_intel.yml`](https://github.com/Cambridge-ICCS/FTorch/blob/main/.github/workflows/test_suite_windows_cpu_intel.yml))
for more information, as this provides another example of how to build and run
FTorch and its integration tests.

If using powershell the setvars and build commands become:
```
cmd /k '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'
cmake -G "NMake Makefiles" -DCMAKE_PREFIX_PATH="C:\Users\<path-to-libtorch-download>\libtorch" -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
cmake --install .
```


### Apple Silicon

FTorch can successfully be built on Apple Silicon machines, including utilising the MPS backend,
following the [regular CMake instructions](|page|/installation/general.html).

To leverage MPS include the `-DGPU_DEVICE=MPS`
[CMake flag](|page|/installation/general.html#cmake-build-options) at build time.


### Conda

Conda is not our preferred approach for managing dependencies, but for users who want
an environment to build FTorch in we provide guidance and environment files in a
[`conda/`](https://github.com/Cambridge-ICCS/FTorch/tree/main/conda) directory.
Note that these environments are not minimal and will install Python, PyTorch,
and other modules required for running the tests and examples.


### GitHub Codespace

It is possible to try FTorch through an interactive browser session without
installing anything locally using GitHub Codespace.
Full instructions are in the
[`codespace/`](https://github.com/Cambridge-ICCS/FTorch/tree/main/codespace) directory.
