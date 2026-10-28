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

4. Batching
    - Demonstrates how to use a PyTorch model trained on 1D vectors to perform inference
      on batched and higher-dimensional data from Fortran.

5. MultiIO
    - Considers a variant of the SimpleNet example which has multiple input
      tensors and multiple output tensors.

6. Looping
    - In most applications we do not just call a net once, but many times.
      To do this in a computationally efficient manner requires some thought.
      This example demonstrates how to structure code in these cases, separating reading
      in of a net from the call to the forward pass.

7. MultiGPU
    - Revisits the SimpleNet example but considering multiple GPUs.

8. MPI
   - Demonstrates how to use FTorch in the context of an MPI program.

9. Autograd
    - **This example is currently under development.** Eventually, it will
      demonstrate automatic differentiation in FTorch by leveraging PyTorch's
      Autograd module.

10. Optimizers
    - Demonstrates how to use FTorch's interface to PyTorch optimizers.

11. Training
    - End-to-end training of a neural net from Fortran using FTorch.

12. Permutation
    - Demonstrates permuting tensor dimensions with `permute_dims` and how
      Fortran's column-major memory layout relates to Torch's row-major layout.

13. GPUPointer
    - Explores whether a `torch_tensor` can be constructed directly on the GPU
      by passing a GPU device pointer to `torch_tensor_from_blob`, using an
      array defined with CUDA Fortran. Requires an NVIDIA GPU, a CUDA Fortran
      capable compiler, and an FTorch build with `-DGPU_DEVICE=CUDA`.

To run select examples as integration tests, use the CMake argument
```
-DCMAKE_BUILD_TESTS=TRUE
```
when building and then call `ctest`. Note that testing is not currently set up
for the MultiGPU or Looping example.
