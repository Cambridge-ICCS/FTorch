title: Online training

[TOC]

## Current state

FTorch has supported offline training of ML models for some time. We are
currently working on extending its functionality to support online training,
too. This will involve exposing the automatic differentiation and
back-propagation functionality in PyTorch/LibTorch.

In the following, we document a workplan of the related functionality. Each step
below will be updated upon completion.

### Operator overloading

Mathematical operators involving Tensors are overloaded, so that we can compute
expressions involving outputs from one or more ML models. For more information
on this, see the [tensor API][pages/tensor.html] documentation page.

For a concrete example of how to compute mathematical expressions involving
Torch tensors, see the
[autograd worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/7_Autograd).

### The `requires_grad` property

For Tensors that you would like to differentiate with respect to, be sure to
set the `requires_grad` optional argument to `.true.` when you construct it.

### The `backward` operator

*Not yet implemented.*
