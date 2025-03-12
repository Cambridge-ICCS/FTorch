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
expressions involving outputs from one or more ML models.

Whilst it's possible to import such functionality with a bare
```fortran
use ftorch
```
statement, the best practice is to import specifically the operators that you
wish to use. Note that the assignment operator `=` has a slightly different
notation:
```fortran
use ftorch, only: assignment(=), operator(+), operator(-), operator(*), &
  operator(/), operator(**)
```

If you would like to make use of scalar multiplication or scalar division, this
can be achieved by setting the scalar as a rank-1 `torch_tensor` with a single
entry. For example:
```fortran
call torch_tensor_from_array(multiplier, [3.0_wp], [1], torch_kCPU)
```

For a concrete example of how to compute mathematical expressions involving
Torch tensors, see the associated
[worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/7_Autograd).

### The `requires_grad` property

For Tensors that you would like to differentiate with respect to, be sure to
set the `requires_grad` optional argument to `.true.` when you construct it.

### The `backward` operator

*Not yet implemented.*
