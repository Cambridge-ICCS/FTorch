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

To build in native Windows using Visual Studio and the Intel Fortran resources the following additional dependencies are required:

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

The following is an example `cmd` script that installs FTorch and runs the integration tests. It assumes you have already
installed CMake, git, the Intel compilers, and Visual Studio.

```cmd
rem disable output for now
ECHO ON

rem load intel compilers
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

rem download ftorch
git clone https://github.com/Cambridge-ICCS/FTorch.git
cd FTorch

rem make venv
python -m venv .ftorch

rem activate the environment
call .ftorch\Scripts\activate

rem install torch
pip install torch torchvision torchaudio

rem enable output
ECHO ON

rem run cmake to generate build scripts
rem (update CMAKE_PREFIX_PATH depending on location of ftorch venv)
cmake -Bbuild -G "NMake Makefiles" -DCMAKE_Fortran_FLAGS="/fpscomp:logicals" ^
 -DCMAKE_PREFIX_PATH="C:\Users\Quickemu\Downloads\FTorch\.ftorch\Lib\site-packages" ^
 -DCMAKE_BUILD_TYPE=Release ^
 -DCMAKE_BUILD_TESTS=True ^
 -DCMAKE_Fortran_COMPILER=ifx -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx

rem build and install ftorch
cmake --build build
cmake --install build

rem quit if this raises an error
if %errorlevel% neq 0 exit /b %errorlevel%

ECHO OFF
rem add ftorch and pytorch libs to path
rem (update these depending on where you installed ftorch and where you created the venv)
set PATH=C:\Users\Quickemu\Downloads\FTorch\.ftorch\Lib\site-packages;%PATH%
set PATH=C:\Program Files (x86)\FTorch\bin;%PATH%
set PATH=C:\Users\Quickemu\Downloads\FTorch\.ftorch\Lib\site-packages\torch\lib;%PATH%

cd ..

rem run integration tests
ECHO ON
run_integration_tests.bat
if %errorlevel% neq 0 exit /b %errorlevel%
```

We recommend Windows users review the Windows CI workflow (`.github/workflows/test_suite_windows.yml`) for more
information, as this provides another example of how to build and run FTorch and its integration tests.

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
