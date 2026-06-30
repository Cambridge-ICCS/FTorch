# Testing

This subdirectory provides automated tests for the FTorch library.

## Pre-requisites for running tests

* FTorch
* [pFUnit](https://github.com/Goddard-Fortran-Ecosystem/pFUnit) (it is not necessary to build this with MPI support at the moment for these tests).

## Building and running tests

From inside the `tests/` directory

    mkdir build
    cd build
    cmake ..
    make

## Unit tests

Below we summarise what is tested in each unit test file (with a CPU device
assumed unless otherwise stated):

* `unittest_tensor_autograd.pf`: Core automatic differentiation functionality,
  e.g., behaviour of `torch_tensor_zero_grad`, `torch_tensor_requires_grad`, and
  the `retain_graph` argument.
* `unittest_tensor_constructors_destructors.pf`: Procedures for constructing and
  destroying `torch_tensor`s, namely `torch_tensor_empty`, `torch_tensor_zeros`,
  `torch_tensor_ones`, `torch_tensor_from_array`, `torch_tensor_from_blob`,
  `torch_tensor_delete`.
* `unittest_tensor_interrogation.pf`: Procedures that interrogate
  `torch_tensor`s, namely `torch_tensor_get_rank`, `torch_tensor_get_shape`,
  `torch_tensor_get_stride`, `torch_tensor_get_dtype`,
  `torch_tensor_get_device_type`, `torch_tensor_get_device_index`, and
  `torch_tensor_requires_grad`.
* `unittest_tensor_interrogation_cuda.pf`: As for the above but with the CUDA
  device type on a GPU.
* `unittest_tensor_manipulation.pf`: Procedures that manipulate `torch_tensor`s,
  namely `torch_tensor_zero`.
* `unittest_tensor_manipulation_cuda.pf`: As for the above but with the CUDA
  device type on a GPU.
* `unittest_tensor_operator_overloads.pf`: Operator overloads for
  `torch_tensor`s, e.g., mathematical operators like `+`, `-`, `*`, `/`, `**`.
* `unittest_tensor_operator_overloads_autograd.pf`: Test that derivatives of the
  `torch_tensor` operator overloads in the above can be computed correctly.
* `unittest_tensor_operators.pf`: Operators acting on `torch_tensor`s, namely
  the reduction operators `torch_tensor_sum` and `torch_tensor_mean`.
* `unittest_tensor_operators_autograd.pf`: Test that derivatives of the
  `torch_tensor` operators in the above can be computed correctly.
