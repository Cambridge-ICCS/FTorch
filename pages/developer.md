title: Developer Guide

If you would like to contributre to the FTorch project, or modify the code at a deeper level, please see below for guidance.

[TOC]

## Getting involved

## Fortran source and fypp

## git hook

## Extending the API

If you have a Torch functionality that you wish to bring in from the C++ API to the FTorch Fortran API the steps are generally as follows:

* Modify `ctorch.cpp` to create a version of the function.
* Add the function to the header file `ctorch.h`
* Modify `ftorch.fypp` to create a fortran version of the function
  that binds to the version in `ctorch.cpp`.

Details of C++ functionalities available to be wrapped can be found
in the [libtorch C++ API](https://pytorch.org/cppdocs/).

As this is an open-source project we appreciate any contributions
back from users that have extended the functionality.
If you have done something but don't know where to start with
open-source contributions please get in touch!

### General guidelines

* Match optional argument defaults between Fortran, C, and C++<br>
  Principle of Least Surprise.

## Documentation

The API documentation for FTorch is generated using FORD.
