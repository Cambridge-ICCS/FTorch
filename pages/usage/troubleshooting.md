title: Troubleshooting
author: Jack Atkinson
date: Last Updated: October 2025

## FAQ

If you are experiencing problems building or using FTorch please see below for guidance on common problems or queries.

- [Usage](#usage)
- [Common Errors](#common-errors)
    - [No specific subroutine](#no-specific-subroutine)
    - [Segmentation faults](#segmentation-faults)


### Usage

##### Why are inputs/outputs to/from torch models arrays?

The reason input and output tensors to/from [[ftorch(module):torch_model_forward(subroutine)]]
are contained in arrays is because it is possible to pass multiple input tensors to
the `forward()` method of a torch net, and it is possible for the net to return
multiple output tensors.

The nature of Fortran means that it is not possible to set an arbitrary number
of inputs to the [[ftorch(module):torch_model_forward(subroutine)]] subroutine,
so instead we use a single array of input tensors which _can_ have an arbitrary length.
Similarly, a single array of output tensors is used.

Note that this does not refer to batching data.
This should be done in the same way as in Torch; by extending the dimensionality of
the input tensors.

##### Do I need to set `torch.inference_mode()`, `torch.no_grad()`, or `torch.eval()` somewhere like in PyTorch?

By default we disable gradient calculations for tensors and models and place models in
evaluation mode for efficiency.
These can be adjusted using the `requires_grad` and `is_training` optional arguments
in the Fortran interface. See the [API procedures documentation](|url|/lists/procedures.html)
for [[ftorch(module):torch_tensor_from_array(interface)]] and
[[ftorch(module):torch_model_load(subroutine)]] etc. for details.


### Common Errors

#### No specific subroutine

FTorch makes heavy use of Fortran `interfaces` to `module procedure`s to achieve
[overloading of subroutines](https://fortran-lang.org/learn/oop_features_in_fortran/object_based_programming_techniques/)
such that for users do not need to call a different subroutine for each rank or type
of tensor.

If you make a call to a subroutine that fails to match anything in the interface
you will face a compile-time error of the form:
```
   42 |   call torch_tensor_from_array(tensor, in_data, tensor_layout, torch_kCPU)
      |                                                                          1
Error: There is no specific subroutine for the generic ‘torch_tensor_from_array’ at (1)
```

The first thing to do in this instance is to inspect the interface you are trying to
call, and instead attempt to call the specific procedure you expect to use.
This can often provide more instructive error messages about what you are doing
incorrectly

##### `int64` versions of `ftorch` for large tensors

An alternative cause of the 'no specific subroutine' error can occur if your tensor
dimension is larger than FTorch supports by default.
Currently FTorch represents the number of elements in an array dimension using
32-bit integers. For most users this will be more than enough, but if your code
uses large tensors (where large means more than 2,147,483,647 elements
in any one dimension (the maximum value of a 32-bit integer)), you may you may
need to compile `ftorch` with 64-bit integers. If you do not, you may receive a
compile time error like the following:

To fix this, rebuild FTorch with 64-bit integers by modifying the following line in
`src/ftorch.fypp`
```fortran
integer, parameter :: ftorch_int = int32 ! set integer size for FTorch library
```
to instead use 64-bit integers:
```fortran
integer, parameter :: ftorch_int = int64 ! set integer size for FTorch library
```
Note: _You will need to re-run `fypp` to regenerate the source files as described in the 
[developer documentation](|page|/developer/developer.html)_


#### Segmentation faults

##### Missing import for overloaded assignment operator

Whenever you execute code involving [[ftorch(module):torch_tensor(type)]]s on each side
of an equals sign, the overloaded assignment operator should be triggered.
As such, if you aren't using the bare `use ftorch` import then you should ensure you
specify `use ftorch, only: assignment(=)` (as well as any other module members you
require). See the [tensor documentation](|page|/usage/tensor.html) for more details.
