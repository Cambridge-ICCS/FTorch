# FTorch: Examples

This directory contains a number of examples of how to use the library:

1. SimpleNet
    - Starting point for new users of the library
    - Most basic example showing how to take a PyTorch net, save it, and couple it to a Fortran code.

2. ResNet-18
    - More complex example demonstrating how to use the library with a multidimensional input.
    - Convert a pre-trained model to TorchScript and call from Fortran.

3. MultiGPU
	- Revisits the SimpleNet example but considering multiple GPUs.

To run the examples as integration tests, use the CMake argument
```
-DCMAKE_BUILD_TESTS=TRUE
```
when building and then call `ctest`. Note that testing is not currently set up
for the MultiGPU example.
