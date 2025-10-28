title: Troubleshooting

If you are experiencing problems building or using FTorch please see below for guidance on common problems.

[TOC]

## FAQ

### Why are inputs/outputs to/from torch models arrays?

The reason input and output tensors to/from [[ftorch(module):torch_model_forward(subroutine)]]
are contained in arrays is because it is possible to pass multiple input tensors to
the `forward()` method of a torch net, and it is possible for the net to return
multiple output arrays.<br>
The nature of Fortran means that it is not possible to set an arbitrary number
of inputs to the [[ftorch(module):torch_model_forward(subroutine)]] subroutine,
so instead we use a single array of input tensors which _can_ have an arbitrary length.
Similarly, a single array of output tensors is used.

Note that this does not refer to batching data.
This should be done in the same way as in Torch; by extending the dimensionality of
the input tensors.

### Common sources of segmentation faults

#### 1. Missing import for overloaded assignment operator

Whenever you execute code involving [[ftorch(module):torch_tensor(type)]]s on each side
of an equals sign, the overloaded assignment operator should be triggered.
As such, if you aren't using the bare `use ftorch` import then you should ensure you
specify `use ftorch, only: assignment(=)` (as well as any other module members you
require). See the [tensor documentation](|page|/usage/tensor.html) for more details.

### Do I need to set `torch.inference_mode()`, `torch.no_grad()`, or `torch.eval()` somewhere like in PyTorch?

By default we disable gradient calculations for tensors and models and place models in
evaluation mode for efficiency.
These can be adjusted using the `requires_grad` and `is_training` optional arguments
in the Fortran interface. See the [API procedures documentation](|url|lists/procedures.html)
for details.

### How do I compile an int64 version of `ftorch` for large tensors?

Currently FTorch represents the number of elements in an array dimension using
32-bit integers. For most users this will be more than enough, but if your code
uses large tensors (where large means more than 2,147,483,647 elements
in any one dimension (the maximum value of a 32-bit integer)), you may you may
need to compile `ftorch` with 64-bit integers. If you do not, you may receive a
compile time error like the following:
```
   39 |   call torch_tensor_from_array(tensor, in_data, tensor_layout, torch_kCPU)
      |                                                                          1
Error: There is no specific subroutine for the generic ‘torch_tensor_from_array’ at (1)
```

To fix this, we can build ftorch with 64-bit integers. We need to modify this
line in `src/ftorch.fypp`
```fortran
integer, parameter :: ftorch_int = int32 ! set integer size for FTorch library
```

We can use 64-bit integers by changing the above line to this
```fortran
integer, parameter :: ftorch_int = int64 ! set integer size for FTorch library
```

Finally, you will need to run `fypp` (`fypp` is not a core dependency, so you
may need to install this separately) e.g.,
```bash
fypp src/ftorch.fypp src/ftorch.F90
```
