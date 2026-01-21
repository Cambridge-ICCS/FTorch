title: Worked Examples
author: Joe Wallwork
date: Last Updated: October 2025

## Worked Examples

The FTorch repository comes with a number of documented
[worked examples](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples).

These are designed to introduce users to FTorch and how to use the various features.

A subset of the examples are used as integration tests as part of FTorch's
[test suite](|page|/developer/testing.html).

#### 1) Tensor

[This worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/1_Tensor)
provides a simple demonstration of how to create, manipulate,
interrogate, and destroy instances of the `torch_tensor` derived type. This is
one of the core derived types in the FTorch library, providing an interface to
the `torch::Tensor` C++ class. Like `torch::Tensor`, the `torch_tensor` derived
type is designed to have a similar API to PyTorch's `torch.Tensor` class.

#### 2) SimpleNet

[This worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/2_SimpleNet)
provides a simple but complete demonstration of how to use the library.
It uses simple PyTorch 'net' that takes an input vector of length 5 and applies a single
Linear layer to multiply it by 2.
The aim is to demonstrate the most basic features of coupling before worrying about
more complex issues that are covered in later examples.

#### 3) Resnet

[This worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/3_ResNet)
provides a more realistic demonstration of how to use the library,
using ResNet-18 to classify an image.
As the input to this model is four-dimensional (batch size, colour, x, y),
care must be taken dealing with the data array in Python and Fortran.
See [when to transpose arrays](|page|/usage/transposing.html) for more details.

#### 4) MultiIO

[This worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/4_MultiIO)
considers a variant of the SimpleNet demo, which demonstrates how to account for
multiple input tensors and multiple output tensors.

#### 5) Looping

[This worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/5_Looping)
demonstrates best practices for performing inference on the same network with
different input multiple times in the same workflow.

#### 6) MultiGPU

[This worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/6_MultiGPU)
builds on the SimpleNet demo and shows how to account for the case of sending different
data to multiple GPU devices.

#### 7) MPI
[This worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/7_MPI)
demonstrates how to run the SimpleNet example in the context of MPI parallelism,
running the net with different input arrays on each MPI rank.

#### 8) Autograd

[These worked examples](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/8_Autograd)
demonstrate how to perform automatic differentiation of operations available in
FTorch by leveraging PyTorch's Autograd module. The first example shows how to
differentiate through mathematical expressions involving Torch tensors, while
the second example shows how to differentiate propagation through a simple
neural network.
