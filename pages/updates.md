title: Recent API Changes

## February 2025

If you use a version of FTorch from before commit
[c85185e](c85185e6c261606c212dd11fee734663d610b695)
(February 2025) you will notice that the main `CMakeLists.txt` file has moved
from `src/` to the root level of the FTorch repository. This move was mainly to
simplify the development experience, such that the examples could be built as
integration tests as part of FTorch, without needing to copy the examples into
a subdirectory of `src/` (as was done previously). For consistency, the other
tests have also been moved from `src/tests/` to `tests/`, with the
`run_test_suite` scripts updated appropriately.

The only difference most users should need to take account of is that the build
directory should no longer be within `src/`. Instead, simply create the build
directory in the root level of the FTorch repository. For example:
```sh
cd /path/to/FTorch
rm -rf build
mkdir build
cd build
cmake .. <CMAKE_ARGUMENTS>
```

## January 2025

If you use a version of FTorch from before commit
[c488f20](c488f20d8d49a15f98176c39a6c8e8db8e708f51)
(January 2025) you may notice that the `device_type` argument for
`torch_model_load` changed from being optional to being compulsory. This is
because the optional argument defaulted to `torch_kCPU`, which is not suitable
for GPU workloads. For recent FTorch configurations, simply specify the device
type as the third argument. For example:
```fortran
type(torch_module) :: model
character(len=17), parameter :: filename = "my_saved_model.pt"
model = torch_module_load(model, filename, torch_kCPU)
```

## June 2024

If you use a version of FTorch from before commit
[e92ad9e](https://github.com/Cambridge-ICCS/FTorch/commit/e92ad9ec7c2198dbb2ca819854d604b984d293c4)
(June 2024) you will notice that the latest API documentation is not suitable.
This is because a number of breaking changes were made to the FTorch API in preparation
for implementing new functionalities.

This page describes how to migrate from code (pre-e92ad9e) to the most recent version.

If you are already using a more recent version there is no need to read this page.

[TOC]

### Why?

We realise that this forms an inconvenience to those of you who are actively
using FTorch and is not something we did lightly.
These changes were necessary to improve functionality and we have made them in one go
as we move towards a stable API and first release in the very near future.
Once the first release is set then the API becomes standardised then changes like this
will be avoided. We hope that this is the last time we have such a shift.

The changes allow us to implement two new features:

1. Multiple output tensors
   Previously you could pass an array of several input tensors to a Torch model, but
   only receive a single output tensor back. Now you can use models that return several
   output tensors by passing an array of output tensors instead.
2. Preparation for autograd functionality
   We hope to make it easier to access the autograd features of PyTorch from within Fortran.
   To do this we needed to change how data was assigned from a Fortran array to a Torch tensor.
   This is now done via a subroutine call rather than a function.

<br>

### Changes and how to update your code

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
call torch_model_load(model, 'path_to_saved_net.pt', torch_kCPU)
call torch_model_forward(model, in_tensors, out_tensors)
```
Note that the `device_type` argument has also been specified in the call to
`torch_model_load`, for the reason mentioned [above](#january-2025).

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
