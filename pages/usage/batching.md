title: Batching
author: Jack Atkinson
date: Last Updated: January 2026

## Batching in FTorch

Batching in FTorch works much the same as it does in PyTorch;
if your model supports batched inference in Python, it will do so in
Fortran via FTorch with no special changes required.

To leverage batching simply add a leading batch dimension to your (Fortran) input
arrays, and FTorch will apply the model independently to each batch element,
preserving the batch structure in the output.

### Example

See the [Batching worked example](|page|/usage/worked_examples.html)
for a complete demonstration.
This illustrates how to use a model trained on 1D vectors for batched and
higher-dimensional inference, both in Python and Fortran, and highlights common
pitfalls.

The following code snippet illustrates the process of using the same net for both
single and batched inference by adding batching dimensions to the front of the Fortran
arrays before casting to torch tensors:

```fortran
  use, intrinsic :: iso_fortran_env, only : sp => real32
  use ftorch, only : torch_model, torch_tensor, torch_kCPU, torch_delete, &
                     torch_tensor_from_array, torch_model_load, torch_model_forward

  ! Single input and output
  real(sp), dimension(5), target   :: in_data_single, out_data_single
  type(torch_tensor), dimension(1) :: in_tensors_single, out_tensors_single

  ! Multidimensional batched input and output
  real(sp), dimension(2,3,5), target :: in_data_batch, out_data_batch
  type(torch_tensor), dimension(1)   :: in_tensors_batch, out_tensors_batch

  ! Load a single torch model to be used for both regular and batched inference
  call torch_model_load(model, "path/to/saved/model.pt", torch_kCPU)

  ! Regular inference on a single input
  call torch_tensor_from_array(in_tensors_single(1), in_data_single, torch_kCPU)
  call torch_tensor_from_array(out_tensors_single(1), out_data_single, torch_kCPU)

  call torch_model_forward(model, in_tensors_single, out_tensors_single)
  
  call torch_delete(in_tensors_single)
  call torch_delete(out_tensors_single)

  ! Multidimensional batched inference
  call torch_tensor_from_array(in_tensors_batch(1), in_data_batch, torch_kCPU)
  call torch_tensor_from_array(out_tensors_batch(1), out_data_batch, torch_kCPU)

  call torch_model_forward(model, in_tensors_batch, out_tensors_batch)

  call torch_delete(in_tensors_batch)
  call torch_delete(out_tensors_batch)

  ! Delete model after using for both single and batched inference
  call torch_delete(model)
```

Batching for nets with multiple inputs and outputs is also supported, provided the
batching dimensions are the same for all inputs (and outputs) and the net architecture
is designed in a way that does not interfere with batching (see below).
This can be explored as an extension to the
[Multiple input/output worked example](|page|/usage/worked_examples.html).

### Key Points

There are a few key points to be aware of when extending your code to make use of
batching. These are also true in PyTorch, but we repeat them here for reinforcement:

- **Shape matters:** The last dimension of your input array _must_ match
  the model’s expected feature size. Any number of leading batch
  (or time) dimensions are supported.
- **Multiple inputs:** If your model takes multiple input tensors, all
  must have the same batch dimensions.
- **Output:** The output shape must mirror the input batch structure.
  This is also true in PyTorch, but in Fortran we need to specify the dimension of our
  output arrays in advance.
- **Error handling:** Shape mismatches or incorrect batch placement
  will result in matrix multiplication runtime errors, as in PyTorch.
- **Architectural gotchas:** Avoid models that flatten, permute, or concatenate inputs
  in a way that destroys the batch dimension, or that mix batch and feature dimensions
  incorrectly, just as with PyTorch. If in doubt test batching in PyTorch before trying
  it with FTorch.
