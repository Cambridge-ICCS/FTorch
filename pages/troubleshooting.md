title: Troubleshooting

If you are experiencing problems building or using FTorch please see below for guidance on common problems.

[TOC]

## Windows

If possible we recommend using the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/) (WSL) to build
the library. In this case the build process is the same as for a Linux environment.

If you need to build in native Windows please read the following information:

### Visual Studio

It is possible to build using Visual Studio and the Intel Fortran Compiler. In this case you must install the following:

* [Visual Studio](https://visualstudio.microsoft.com/) ensuring C++ tools are selected and installed.
* [Intel OneAPI Basetoolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
* [Intel OneAPI HPC toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit.html) ensuring that the Intel Fortran compiler and VS integration is selected.

You will then need to load the intel Fortran compilers using `setvars.bat` which is found in the Intel compiler install
directory (see the [intel
docs](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2023-2/use-the-setvars-script-with-windows.html))
for more details.<br>

From `cmd` this can be done with:
```
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```

Finally you will need to add `-G "NMake Makefiles"` to the `cmake` command in the
[regular install instructions](doc/page/cmake.html).<br>
So the basic command to build from `cmd` becomes:
```
cmake -G "NMake Makefiles" -DCMAKE_PREFIX_PATH="C:\Users\<path-to-libtorch-download>\libtorch" -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
cmake --install .
```

The following is an example `cmd` script that installs FTorch and runs the integration tests. It assumes you have already
install `cmake`, `git`, the intel compilers and visual studio.

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

We would also recommend Windows users to review the Windows CI workflow (`.github/workflows/test_suite_windows.yml`) for more
information, as this provides another example of how to build and run FTorch and its integration tests.

If using powershell the setvars and build commands become:
```
cmd /k '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'
cmake -G "NMake Makefiles" -DCMAKE_PREFIX_PATH="C:\Users\<path-to-libtorch-download>\libtorch" -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
cmake --install .
```

### MinGW

It may be tempting to build on Windows using MinGW.
However, [LibTorch does not currently support MinGW](https://github.com/pytorch/pytorch/issues/15099).
Instead please build using Visual Studio and the intel Fortran compiler (ifort) as
detailed in the project README.

## Apple Silicon

At the time of writing, LibTorch is currently only officially available for x86
architectures (according to [pytorch.org](https://pytorch.org/)).
However, the version of PyTorch provided by pip install provides an ARM binary
for LibTorch which works on Apple Silicon.
Therefore you should `pip install torch` in this situation and follow the guidance
on locating Torch within a virtual environment (venv) for CMake.

## FAQ

### Why are inputs/outputs to/from torch models arrays?

The reason input and output tensors to/from [[torch_model_forward(subroutine)]] are
contained in arrays is because it is possible to pass multiple input tensors to
the `forward()` method of a torch net, and it is possible for the net to return
multiple output arrays.<br>
The nature of Fortran means that it is not possible to set an arbitrary number
of inputs to the `torch_model_forward` subroutine, so instead we use a single
array of input tensors which _can_ have an arbitrary length. Similarly, a single
array of output tensors is used.

Note that this does not refer to batching data.
This should be done in the same way as in Torch; by extending the dimensionality of
the input tensors.

### Do I need to set torch.no_grad() or torch.eval() somewhere like in PyTorch?

By default we disable gradient calculations for tensors and models and place models in
evaluation mode for efficiency.
These can be adjusted using the `requires_grad` and `is_training` optional arguments
in the Fortran interface. See the [API procedures documentation](lists/procedures.html)
for details.

### How do I compile an int64 version of `ftorch` for large tensors?

Currently FTorch represents the number of elements in an array dimension using
32-bit integers. For most users this will be more than enough, but if your code
uses large tensors (where large means more than 2,147,483,647 elements
in any one dimension (the maximum value of a 32-bit integer)), you may you may
need to compile `ftorch` with 64-bit integers. If you do not, you may receive a
compile time error like the following:
```
   39 |   call torch_tensor_from_array(tensor, in_data, tensor_layout, torch_kCPU)
      |                                                                          1
Error: There is no specific subroutine for the generic ‘torch_tensor_from_array’ at (1)
```

To fix this, we can build ftorch with 64-bit integers. We need to modify this
line in `src/ftorch.fypp`
```fortran
integer, parameter :: ftorch_int = int32 ! set integer size for FTorch library
```

We can use 64-bit integers by changing the above line to this
```fortran
integer, parameter :: ftorch_int = int64 ! set integer size for FTorch library
```

Finally, you will need to run `fypp` (`fypp` is not a core dependency, so you
may need to install this separately) e.g.,
```bash
fypp src/ftorch.fypp src/ftorch.F90
```
