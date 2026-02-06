title: Tensor API
author: Joe Wallwork
date: Last Updated: October 2025

## Tensor API Documentation

- [Overview](#overview)
- [Procedures](#procedures)
    - [Constructors](#constructors)
    - [Interrogation](#interrogation)
    - [Deallocation](#deallocation)
    - [Manipulation](#manipulation)
    - [Operator Overloading](#operator-overloading)

### Overview

FTorch provides a [[ftorch(module):torch_tensor(type)]] derived type, which exposes the
functionality of the `torch::Tensor` C++ class. The interface is designed to be familiar to
Fortran programmers, whilst retaining strong similarity with `torch::Tensor` and
the `torch.Tensor` Python class.

Under the hood, the [[ftorch(module):torch_tensor(type)]] type holds a pointer to a
`torch::Tensor` object in C++ (implemented using `c_ptr` from the `iso_c_binding`
intrinsic module). This allows us to avoid unnecessary data copies between C++ and
Fortran.

### Procedures

#### Constructors

We provide several subroutines for constructing [[ftorch(module):torch_tensor(type)]]
objects. These include:

* [[ftorch(module):torch_tensor_empty(subroutine)]] which allocates memory for the
  [[ftorch(module):torch_tensor(type)]] but does not set any values.
* [[ftorch(module):torch_tensor_zeros(subroutine)]] which creates a
  [[ftorch(module):torch_tensor(type)]] whose values are uniformly zero.
* [[ftorch(module):torch_tensor_ones(subroutine)]] which creates a
  [[ftorch(module):torch_tensor(type)]] whose values are
  uniformly one.
* [[ftorch(module):torch_tensor_from_array(interface)]] which allows the user to create
  a [[ftorch(module):torch_tensor(type)]] with the same rank, shape, and data type as a
  given Fortran array. Note that the data is *not* copied - the tensor data points to the
  Fortran array, meaning the array must have been declared with the `target` property.
  The array will continue to be pointed to even when operations are applied to the
  tensor, so this subroutine can be used 'in advance' to set up an array for
  outputting data. [[ftorch(module):torch_tensor_from_array(interface)]] may be called
  with or without the `layout` argument - an array which specifies the order in which
  indices should be looped over. The default `layout` is `[1,2,...,n]` implies that data
  will be read into the same indices by Torch. (See the
  [transposing user guide page](|page|/usage/transposing.html) for more details.

It is *compulsory* to call one of the constructors before interacting with it in
any of the ways described in the following. Each of the constructors sets the
pointer attribute of the [[ftorch(module):torch_tensor(type)]]; without this being set,
most of the other operations are meaningless.


#### Interrogation

We provide several subroutines for interrogating [[ftorch(module):torch_tensor(type)]]
objects. These include:

* [[ftorch(module):torch_tensor_get_rank(function)]] which determines the rank
  (i.e., dimensionality) of the tensor.
* [[ftorch(module):torch_tensor_get_shape(function)]] which determines the shape
  (i.e., extent in each dimension) of the tensor.
* [[ftorch(module):torch_tensor_get_dtype(function)]] which determines the data type
  of the tensor in terms of the enums `torch_kInt8`, `torch_kFloat32`, etc.
* [[ftorch(module):torch_tensor_get_device_type(function)]], which determines the device
  type that the tensor resides on in terms of the enums `torch_kCPU`, `torch_kCUDA`,
  `torch_kXPU`, etc.
* [[ftorch(module):torch_tensor_get_device_index(function)]], which determines the
  index of the device that the tensor resides on as an integer.
  For a CPU device, this index should be set to `-1` (the default).
  For GPU devices, the index should be non-negative (defaulting to `0`).

Procedures for interrogation are implemented as methods as well as stand-alone
procedures. For example, `tensor%get_rank` can be used in place of
[[ftorch(module):torch_tensor_get_rank(function)]], omitting the first argument
(which would be the tensor itself). The naming pattern is similar for the other methods
(simply drop the preceding `torch_tensor_`).


#### Deallocation

We provide a subroutine for deallocating the memory associated with a
[[ftorch(module):torch_tensor(type)]] object:
[[ftorch(module):torch_tensor_delete(subroutine)]]. An interface
[[ftorch(module):torch_delete(interface)]] is provided such that this can also
be applied to arrays of tensors.

Manually deallocating a tensor that you declared and constructed in your code is
actually optional. If the tensor was declared in a subroutine then
[[ftorch(module):torch_tensor_delete(subroutine)]] will get called as the
finalizer of [[ftorch(module):torch_tensor(type)]] when it goes out of scope. If
the tensor was declared in a program then the finalizer won't get called, but
this is not considered to be an issue, in the same way that in Fortran it
doesn't matter if allocated arrays aren't deallocated at the end of the program.


#### Manipulation

We provide the following subroutines for manipulating the data values associated
with a [[ftorch(module):torch_tensor(type)]] object:

* [[ftorch(module):torch_tensor_zero(subroutine)]] (aliased as class method `torch_tensor%zero`), which
  sets all the data entries associated with a tensor to zero.

@note
For a concrete example of how to construct, interrogate, manipulate, and delete
Torch tensors, see the
[tensor manipulation worked example](|page|/usage/worked_examples.html).
@endnote


#### Operator overloading

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

##### Overloaded assignment operator

Particular care should be taken with the overloaded assignment operator.
Whenever you execute code involving [[ftorch(module):torch_tensor(type)]]s on each side
of an equals sign, the overloaded assignment operator should be triggered.
As such, if you aren't using the bare `use ftorch` import then you should ensure you
specify `use ftorch, only: assignment(=)` (as well as any other module members you
require).

For a straightforward assignment of two [[ftorch(module):torch_tensor(type)]]s
`a` and `b`,
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
using [[ftorch(module):torch_tensor_empty(subroutine)]] under the hood
(inheriting all the properties of the tensors being added[^1]).
Following this, the intermediate tensor is assigned to `c`.
Finally, the finalizer for [[ftorch(module):torch_tensor(type)]] is called for the
intermediate tensor because it goes out of scope.

[^1]: Note: In most cases, these should be the same, so that the operator makes
sense. In the case of the `requires_grad` property, the values might differ, and
the result should be the logical `.and.` of the two values.

Similarly as above, in the case where you have some function `func` that returns
a [[ftorch(module):torch_tensor(type)]], an intermediate
[[ftorch(module):torch_tensor(type)]] will be created, assigned, and
destroyed because the call will have the form
```fortran
a = func()
```

#### Other operators acting on tensors

We have also exposed the operators for taking the sum or mean over the entries
in a tensor, which can be achieved with the subroutines
[[ftorch(module):torch_tensor_sum(subroutine)]] and
[[ftorch(module):torch_tensor_mean(subroutine)]], respectively.

@note
For a concrete example of how to compute mathematical expressions involving Torch
tensors, see the [autograd worked example](|page|/usage/worked_examples.html).
@endnote
