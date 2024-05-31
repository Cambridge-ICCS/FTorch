# Motivation

We want to be able to run ML models directly in Fortran. Initially let's assume that the model has been trained in some other language (say Python). We want to run inference on this model without having to call the Python executable. This should be possible by using the existing ML C/C++ interfaces.


# PyTorch

PyTorch provides a C++ API (aka LibTorch) that enables inference and online training of pre-trained models in a production environment with no Python dependency. A model's journey from Python to C++ is enabled by TorchScript, an intermediate representation of a PyTorch model. A model trained in Python is first converted to a TorchScript module via tracing or annotation and then serialised to a file. The serialised model can then be loaded in C++ using the C++ API. For more information, see [here](https://pytorch.org/tutorials/advanced/cpp_export.html).


# Description

We have created a minimum C wrapper interface of LibTorch (c_torch.h) that allows us to load and infer a TorchScript model. A Fortran interface (mod_torch) has subsequently been built on top of the C interface. For examples using the C++, C and Fortran interfaces to do inference, see the corresponding ts_inference.* file. A Python script is also provided (p2ts.py) that converts a pre-trained ResNet model to TorchScript and serialises it to a file. This model can be used as the input of the ts_infer_* executables for testing.


## Building

To build, follow the instructions in the [main repository README](/README.md).

## Testing

For information on testing, see the [test subdirectory](test/README.md).


### Summary of files

* ctorch.cpp - Wrapper onto PyTorch for C++
* ctorch.h   - Header file for C++ wrapper
* ftorch.f90 - Wrapper onto PyTorch for Fortran
* CMakeLists.txt - Provides the CMake build system configuration
