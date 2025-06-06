title: Tensor API

[TOC]

## Overview

FTorch provides a `torch_tensor` derived type, which exposes the functionality
of the `torch::Tensor` C++ class. The interface is designed to be familiar to
Fortran programmers, whilst retaining strong similarity with `torch::Tensor` and
the `torch.Tensor` Python class.

Under the hood, the `torch_tensor` type holds a pointer to a `torch::Tensor`
object in C++ (implemented using `c_ptr` from the `iso_c_binding` intrinsic
module). This allows us to avoid unnecessary data copies between C++ and
Fortran.

## Procedures

### Constructors

We provide several subroutines for constructing `torch_tensor` objects. These
include:

* `torch_tensor_empty`, which allocates memory for the `torch_tensor`, but does
  not set any values.
* `torch_tensor_zeros`, which creates a `torch_tensor` whose values are
  uniformly zero.
* `torch_tensor_ones`, which creates a `torch_tensor` whose values are
  uniformly one.
* `torch_tensor_from_array`, which allows the user to create a `torch_tensor`
  with the same rank, shape, and data type as a given Fortran array. Note that
  the data is *not* copied - the tensor data points to the Fortran array,
  meaning the array must have been declared with the `target` property. The
  array will continue to be pointed to even when operations are applied to the
  tensor, so this subroutine can be used 'in advance' to set up an array for
  outputting data. `torch_tensor_from_array` may be called with or without the
  `layout` argument - an array which specifies the order in which indices should
  be looped over. The default `layout` is `[1,2,...,n]` implies that data will
  be read into the same indices by Torch. (See the
  [transposing user guide page](pages/transposing.html) for more details.

It is *compulsory* to call one of the constructors before interacting with it in
any of the ways described in the following. Each of the constructors sets the
pointer attribute of the `torch_tensor`; without this being set, most of the
other operations are meaningless.

### Tensor interrogation

We provide several subroutines for interrogating `torch_tensor` objects. These
include:

* `torch_tensor_get_rank`, which determines the rank (i.e., dimensionality) of
  the tensor.
* `torch_tensor_get_shape`, which determines the shape (i.e., extent in each
  dimension) of the tensor.
* `torch_tensor_get_dtype`, which determines the data type of the tensor in
  terms of the enums `torch_kInt8`, `torch_kFloat32`, etc.
* `torch_tensor_get_device_type`, which determines the device type that the
  tensor resides on in terms of the enums `torch_kCPU`, `torch_kCUDA`,
  `torch_kXPU`, etc.
* `torch_tensor_get_device_index`, which determines the index of the device that
  the tensor resides on as an integer. For a CPU device, this index should be
  set to -1 (the default). For GPU devices, the index should be non-negative
  (defaulting to 0).

Procedures for interrogation are implemented as methods as well as stand-alone
procedures. For example, `tensor%get_rank` can be used in place of
`torch_tensor_get_rank`, omitting the first argument (which would be the tensor
itself). The naming pattern is similar for the other methods (simply drop the
preceding `torch_tensor_`).

### Tensor deallocation

We provide a subroutine for deallocating the memory associated with a
`torch_tensor` object: `torch_tensor_delete`. An interface is provided such that
this can also be applied to arrays of tensors. Calling this subroutine manually
is optional as it is called as a destructor when the `torch_tensor` goes out of
scope anyway.

### Tensor manipulation

We provide the following subroutines for manipulating the data values associated
with a `torch_tensor` object:

* `torch_tensor_zero` (aliased as class method `torch_tensor%zero`), which
  sets all the data entries associated with a tensor to zero.

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


#### Overloaded assignment operator

Particular care should be taken with the overloaded assignment operator.
Whenever you execute code involving `torch_tensor`s on each side of an equals
sign, the overloaded assignment operator should be triggered. As such, if you
aren't using the bare `use ftorch` import then you should ensure you specify
`use ftorch, only: assignment(=)` (as well as any other module members you
require).

For a straightforward assignment of two `torch_tensor`s `a` and `b`,
```fortran
b = a
```
the overloaded assignment operator is called once.

For overloaded operators the situation is more complex. Consider the overloaded
addition operator (the same applies for the rest). When we execute the line
```fortran
c = a + b
```
the addition is evaluated first. It is implemented as a Fortran function and its
return value is an *intermediate* tensor. The setup is such that this is created
using `torch_tensor_empty` under the hood (inheriting all the properties of the
tensors being added[^1]). Following this, the intermediate tensor is assigned
to `c`. Finally, the finalizer for `torch_tensor` is called for the intermediate
tensor because it goes out of scope.

Similarly as above, in the case where you have some function `func` that returns
a `torch_tensor`, an intermediate `torch_tensor` will be created, assigned, and
destroyed because the call will have the form
```fortran
a = func()
```

### Other operators acting on tensors

We have also exposed the operators for taking the sum or mean over the entries
in a tensor, which can be achieved with the subroutines `torch_tensor_sum` and
in `torch_tensor_mean`, respectively.

## Examples

For a concrete example of how to construct, interrogate, manipulate, and delete
Torch tensors, see the
[tensor manipulation worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/1_Tensor).

For an example of how to compute mathematical expressions involving Torch
tensors, see the
[autograd worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/6_Autograd).

[^1]: Note: In most cases, these should be the same, so that the operator makes
sense. In the case of the `requires_grad` property, the values might differ, and
the result should be the logical `.and.` of the two values.
