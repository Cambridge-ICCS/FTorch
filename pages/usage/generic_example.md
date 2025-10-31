title: Generic Example
author: Jack Atkinson
date: Last Updated: October 2025

## Generic Example

This page provides abstract guidance to the general process of using FTorch.
It covers the full process of saving a model from PyTorch and coupling it into a Fortran code.
To see this process in context we strongly advise following the
[Worked Examples](|page|/examples/worked_examples.html).

### Overview of the interfacing process

In order to use FTorch users will typically need to follow these steps:

1. Save a PyTorch model as [TorchScript](https://pytorch.org/docs/stable/jit.html).
2. Write Fortran using the FTorch bindings to use the model from within Fortran.
3. Build and compile the code, linking against the FTorch library

These are outlined in detail below.


#### 1. Saving the model as TorchScript

The trained PyTorch model needs to be exported to
[TorchScript](https://pytorch.org/docs/stable/jit.html).
This can be done from within your code using the
[`jit.script`](https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch.jit.script)
or
[`jit.trace`](https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch.jit.trace)
functionalities from within Python.

If you are not familiar with these we provide a tool
[`pt2ts.py`](https://github.com/Cambridge-ICCS/FTorch/blob/main/utils/pt2ts.py)
as part of FTorch which contains an easily adaptable script to save your
PyTorch model as TorchScript.


#### 2. Using the model from Fortran

To use the trained Torch model from within Fortran we need to import the `ftorch`
module and use the binding routines to load the model, convert the data,
and run inference.
A very simple example is given below.

This minimal snippet loads a saved Torch model, creates an input consisting of a
`10x10` matrix of ones, and runs the model to infer output.  
This is for illustrative purposes only, and we recommend following the
[examples](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples)
before writing your own code to fully explore the features.

```fortran
! Import FTorch library for interfacing with PyTorch
use ftorch, only : torch_model, torch_tensor, torch_kCPU, torch_delete, &
                   torch_tensor_from_array, torch_model_load, torch_model_forward

implicit none

! Generate an object to hold the Torch model
type(torch_model) :: model

! Set up array of n_inputs input tensors and array of n_outputs output tensors
! Note: In this example there is only one input tensor (n_inputs = 1) and one
!       output tensor (n_outputs = 1)
integer, parameter :: n_inputs = 1
integer, parameter :: n_outputs = 1
type(torch_tensor), dimension(n_inputs)  :: model_input_arr
type(torch_tensor), dimension(n_outputs) :: model_output_arr

! Set up the model inputs and output as Fortran arrays
real, dimension(10,10), target :: input
real, dimension(5), target     :: output

! Initialise the Torch model to be used
torch_model_load(model, "/path/to/saved/model.pt", torch_kCPU)

! Initialise the inputs as Fortran array of ones
input = 1.0

! Wrap Fortran data as no-copy Torch Tensors
! There may well be some reshaping required depending on the 
! structure of the model which is not covered here (see examples)
call torch_tensor_from_array(model_input_arr(1), input, torch_kCPU)
call torch_tensor_from_array(model_output_arr(1), output, torch_kCPU)

! Run model forward method and Infer
! Again, there may be some reshaping required depending on model design
call torch_model_forward(model, model_input_arr, model_output_arr)

! Write out the result of running the model
write(*,*) output

! Clean up
call torch_delete(model)
call torch_delete(model_input_arr)
call torch_delete(model_output_arr)
```


#### 3. Building the code

The code now needs to be compiled and linked against our installed library.
Here we describe how to do this for two build systems, CMake and make.

##### CMake
If using CMake include the following in the `CMakeLists.txt`
file to find the FTorch installation and link it to the executable.

This can be done by adding the following to the `CMakeLists.txt` file:
```CMake
find_package(FTorch REQUIRED)
target_link_libraries( <executable> PRIVATE FTorch::ftorch )
message(STATUS "Building with Fortran PyTorch coupling")
```
and using the `-DCMAKE_PREFIX_PATH=</path/to/install/location>` flag when running CMake.  

@note
_If you used the `CMAKE_INSTALL_PREFIX` argument when
[building and installing the library](|page|/installation/general.html)
then you should use the same path for `</path/to/install/location>`._
@endnote

##### Make
To build with `make` we need to _include_ the library and _link_ the
executable against it when compiling.

For full details of the flags to set and the linking process see the
[HPC build pages](|page|/installation/hpc.html/#building-projects-and-linking-to-ftorch).

### Running on GPUs

In order to run a model on GPU, two main changes to the above process are required:

1. When saving your TorchScript model, ensure that it is on the GPU.
2. When calling `torch_tensor_from_array` in Fortran, the device for the input
   tensor(s) should be set to `torch_kCUDA`, rather than `torch_kCPU`.

For more information refer to the [GPU Documentation](|page|/installation/gpu.html)
