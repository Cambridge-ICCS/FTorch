# FTorch: Examples

This directory contains a number of examples of how to use the library:

1. Tensor
    - Starting point for new users of the library
    - Demonstration of how to construct, interrogate, and destroy tensors in
      FTorch.

2. SimpleNet
    - Most basic example showing how to take a PyTorch net, save it, and couple it to a Fortran code.

3. ResNet
    - More complex example demonstrating how to use the library with a multidimensional input.
    - Convert a pre-trained model to TorchScript and call from Fortran.

4. MultiIO
    - Considers a variant of the SimpleNet example which has multiple input
      tensors and multiple output tensors.

5. Looping
    - In most applications we do not just call a net once, but many times.
      To do this in a computationally efficient manner requires some thought.
      This example demonstrates how to structure code in these cases, separating reading
      in of a net from the call to the forward pass.

6. MultiGPU
    - Revisits the SimpleNet example but considering multiple GPUs.

7. MPI
   - Demonstrates how to use FTorch in the context of an MPI program.

8. Autograd
    - **This example is currently under development.** Eventually, it will
      demonstrate automatic differentiation in FTorch by leveraging PyTorch's
      Autograd module.

To run select examples as integration tests, use the CMake argument
```
-DCMAKE_BUILD_TESTS=TRUE
```
when building and then call `ctest`. Note that testing is not currently set up
for the MultiGPU or Looping example.
