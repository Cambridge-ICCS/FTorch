title: Online training

[TOC]

## Current state


FTorch has supported offline training of ML models for some time (see the
[offline training user guide page](pages/offline.html) for details). We are
currently working on extending its functionality to support online training,
too. This will involve exposing the backpropagation and optimization
functionalities of PyTorch/LibTorch.

In the following, we document a workplan of the related functionality. Each step
below will be updated upon completion.

### Operator overloading

Mathematical operators involving Tensors are overloaded, so that we can compute
expressions involving outputs from one or more ML models. For more information
on this, see the [tensor API][pages/tensor.html] documentation page.

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
Torch tensors, see the
[autograd worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/8_Autograd).

### The `requires_grad` property

For Tensors that you would like to differentiate with respect to, be sure to
set the `requires_grad` optional argument to `.true.` when you construct it.

### Backpropagation

Having defined some tensors with the `requires_grad` property set to `.true.`
and computed another tensor in terms of an expression involving these, we can
compute gradients of that tensor with respect to those that it depends on. This
is achieved using the `torch_tensor_backward` subroutine. For example, for
input tensors `a` and `b` and an output tensor `Q`:

```fortran
call torch_tensor_from_array(a, in_data1, tensor_layout, torch_kCPU, &
                             requires_grad=.true.)
call torch_tensor_from_array(b, in_data2, tensor_layout, torch_kCPU, &
                             requires_grad=.true.)
call torch_tensor_from_array(Q, out_data1, tensor_layout, torch_kCPU)

Q = a * b

call torch_tensor_backward(Q)
```

Following the example code above, we can extract gradients of `Q` with respect
to `a` and/or `b`. To do this, we can use the `torch_tensor_get_gradient`
subroutine. That is, for tensors `dQda` and `dQdb`:

```fortran
call torch_tensor_from_array(dQda, out_data2, tensor_layout, torch_kCPU)
call torch_tensor_get_gradient(a, dQda)

call torch_tensor_from_array(dQdb, out_data3, tensor_layout, torch_kCPU)
call torch_tensor_get_gradient(b, dQdb)
```

#### `retain_graph` argument

If you wish to call the backpropagation operator multiple times then you may
need to make use of the `retain_graph` argument for `torch_tensor_backward`.
This argument accepts logical values and defaults to `.false.`, for consistency
with PyTorch and LibTorch. According to the
[PyTorch docs](https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html),
`retain_graph=.true.` will not be needed in most cases, but it's useful to have
for the cases where it is.

#### Zeroing gradients

Having computed gradients of one tensor with respect to its dependencies,
suppose you wish to compute gradients of another tensor. Since the gradient
values associated with each dependency are accumulated, you should zero the
gradients before computing the next gradient. This can be achieved using the
`torch_tensor_zero_grad` subroutine.

Following the example code above:

```fortran
Q = a * b
P = a + b

call torch_tensor_backward(Q)

! ...

call torch_tensor_zero_grad(a)
call torch_tensor_zero_grad(b)

call torch_tensor_backward(P, retain_graph=.true.)

! ...
```

#### Extracting gradients

Note that `torch_tensor_get_gradient` must be called after every call to
`torch_tensor_backward` or `torch_tensor_zero_grad`, even if the gradient for
the same tensor is being extracted into the same array. This is due to the way
that pointers are handled on the C++ side.

### Optimisation

*Not yet implemented.*

### Loss functions

*Not yet implemented.*
