title: Recent API Changes

If you use a version of FTorch from before commit 
[e92ad9e](https://github.com/Cambridge-ICCS/FTorch/commit/e92ad9ec7c2198dbb2ca819854d604b984d293c4)
(June 2024) you will notice that the latest API documentation is not suitable.
This is because a number of breaking changes were made to the FTorch API in preparation
for implementing new functionalities.

This page describes how to migrate from code (pre-e92ad9e) to the most recent version.

If you are already using a more recent version there is no need to read this page.

[TOC]

## Why?

We realise that this forms an inconvenience to those of you who are actively
using FTorch and is not something we did lightly.
These changes were neccessary to improve functionality and we have made them in one go
as we move towards a stable API and first release in the very near future.
Once the first release is set then the API becomes standardised then changes like this
will be avoided. We hope that this is the last time we have such a shift.

The changes allow us to implement two new features:

1. Multiple output tensors
   Previously you could pass an array of several input tensors to a Torch model, but
   only recieve a single output tensor back. Now you can use models that return several
   output tensors by passing an array of output tensors instead.
2. Preparation for autograd functionality
   We hope to make it easier to access the autograd features of PyTorch from within Fortran.
   To do this we needed to change how data was assigned from a Fortran array to a Torch tensor.
   This is now done via a subroutine call rather than a function.

<br>

## Changes and how to update your code

<br>

#### `torch_tensor`s are created using a subroutine call, not a function

Previously you would have created a Torch tensor and assigned some fortran data to it as follows:
```fortran
real, dimension(5), target :: fortran_data
type(torch_tensor) :: my_tensor
integer :: tensor_layout(1) = [1]

my_tensor = torch_tensor_from_array(fortran_data, tensor_layout, torch_kCPU)
```
<br>
Now a call is made to a subroutine with the tensor as the first argument:
```fortran
real, dimension(5), target :: fortran_data
type(torch_tensor) :: my_tensor
integer :: tensor_layout(1) = [1]

call torch_tensor_from_array(my_tensor, fortran_data, tensor_layout, torch_kCPU)
```

<br>

#### `module` becomes `model` and loading becomes a subroutine call, not a function

Previously a neural net was referred to as a '`module`' and loaded using appropriately
named functions and types.
```fortran
type(torch_module) :: model
model = torch_module_load(args(1))
call torch_module_forward(model, in_tensors, out_tensors)
```
<br>
Following user feedback we now refer to a neural net and its associated types and calls
as a '`model`'.
The process of loading a net is also now a subroutine call for consistency with the
tensor creation operations:
```fortran
type(torch_model) :: model
call torch_model_load(model, 'path_to_saved_net.pt')
call torch_model_forward(model, in_tensors, out_tensors)

```

<br>

#### `n_inputs` is no longer required

Previously when you called the forward method on a net you had to specify the number of tensors
in the array of inputs:
```fortran
call torch_model_forward(model, in_tensors, n_inputs, out_tensors)
```
<br>
Now all that is supplied to the forward call is the model, and the arrays of input and
output tensors. No need for `n_inputs` (or `n_outputs`)!
```fortran
call torch_model_forward(model, in_tensors, out_tensors)
```

<br>

#### Outputs now need to be an array of `torch_tensor`s

Previously you passed an array of `torch_tensor` types as inputs, and a single `torch_tensor`
to the forward method:
```fortran
type(torch_tensor), dimension(n_inputs) :: input_tensor_array
type(torch_tensor) :: output_tensor
...
call torch_model_forward(model, input_tensor_array, n_inputs, output_tensor)
```
<br>
Now both the inputs and the outputs need to be an array of `torch_tensor` types:
```fortran
type(torch_tensor), dimension(n_inputs)  :: input_tensor_array
type(torch_tensor), dimension(n_outputs) :: output_tensor_array
...
call torch_model_forward(model, input_tensor_array, output_tensor_array)
```
