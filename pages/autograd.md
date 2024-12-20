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
```
use ftorch, only: assignment(=), operator(+), operator(-), operator(*), &
  operator(/), operator(**)
```

For a concrete example of how to compute mathematical expressions involving
Torch tensors, see the associated
[worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/6_Autograd).

### The `requires_grad` property

*Not yet implemented.*

### The `backward` operator

*Not yet implemented.*
