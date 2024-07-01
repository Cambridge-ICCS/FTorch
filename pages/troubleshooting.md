title: Troubleshooting

If you are experiencing problems building or using FTorch please see below for guidance on common problems.

[TOC]

## Windows

If possible we recommend using the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/) (WSL) to build the library.
In this case the build process is the same as for a Linux environment.

If you need to build in native Windows please read the following information:

### Visual Studio

It is possible to build using Visual Studio and the Intel Fortran Compiler  
In this case you must install 

* [Visual Studio](https://visualstudio.microsoft.com/) ensuring C++ tools are selected and installed.
* [Intel OneAPI Basetoolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
* [Intel OneAPI HPC toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit.html) ensuring that the Intel Fortran compiler and VS integration is selected.

You will then need to load the intel Fortran compilers using `setvars.bat`
which is found in the Intel compiler install directory (see the 
[intel docs](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2023-2/use-the-setvars-script-with-windows.html))
for more details.<br>
From CMD this can be done with:
```
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```

Finally you will need to add `-G "NMake Makefiles"` to the `cmake` command in the
[regular install instructions](doc/page/cmake.html).<br>
So the basic command to build from CMD becomes:
```
cmake -G "NMake Makefiles" -DCMAKE_PREFIX_PATH="C:\Users\<path-to-libtorch-download>\libtorch" -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
cmake --install .
```

If using powershell the setvars and build commands become:
```
cmd /k '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'
cmake -G "NMake Makefiles" -DCMAKE_PREFIX_PATH="C:\Users\<path-to-libtorch-download>\libtorch" -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
cmake --install .
```

### MinGW

It may be tempting to build on Windows using MinGW.
However, [libtorch does not currently support MinGW](https://github.com/pytorch/pytorch/issues/15099).
Instead please build using Visual Studio and the intel Fortran compiler (ifort) as
detailed in the project README.

## Apple Silicon

At the time of writing, libtorch is currently only officially available for x86
architectures (according to [pytorch.org](https://pytorch.org/)).
However, the version of PyTorch provided by pip install provides an ARM binary
for libtorch which works on Apple Silicon.
Therefore you should `pip install torch` in this situation and follow the guidance
on locating Torch within a virtual environment (venv) for CMake.

## FAQ

### Why are inputs to torch models an array?

The reason input tensors to [[torch_module_forward(subroutine)]] are contained in an
array is because it is possible to pass multiple input tensors to the `forward()`
method of a torch net.<br>
The nature of Fortran means that it is not possible to set an arbitrary number
of inputs to the `torch_module_forward` subroutine, so instead we use an single array
of input tensors which _can_ have an arbitrary length of `n_inputs`.

Note that this does not refer to batching data.
This should be done in the same way as in Torch; by extending the dimensionality of
the input tensors.

### Do I need to set torch.no_grad() or torch.eval() somewhere like in PyTorch?

By default we disable gradient calculations for tensors and models and place models in
evaluation mode for efficiency.
These can be adjusted using the `requires_grad` and `is_training` optional arguments
in the Fortran interface. See the [API procedures documentation](lists/procedures.html)
for details.
