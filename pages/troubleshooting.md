title: Troubleshooting

If you are experiencing problems building or using FTorch please see below for guidance on common problems.

[TOC]

## Windows

If possible we recommend using the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/) (WSL) to build the library.
In this case the build process is the same as for a Linux environment.

If you need 

### Visual Studio

Use Visual Studio and the Intel Fortran Compiler  
In this case you must install 
* [Visual Studio](https://visualstudio.microsoft.com/)
* [Intel OneAPI Base and HPC toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) (ensure that the Intel Fortran compiler and VS integration is selected).

You should then be able to build from CMD following the 

### MinGW

It may be tempting to build on Windows using MinGW.
However, [libtorch does not currently support MinGW](https://github.com/pytorch/pytorch/issues/15099).
Instead please build using Visual Studio and the intel fortran compiler (ifort) as
detailed in the project README.


