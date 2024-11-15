# Testing

This subdirectory provides automated tests for the FTorch library.

## Pre-requisites for running tests

* FTorch
* [pFUnit](https://github.com/Goddard-Fortran-Ecosystem/pFUnit) (it is not necessary to build this with MPI support at the moment for these tests).

## Building and running tests

From inside the `tests/` directory

    mkdir build
    cd build
    cmake ..
    make
