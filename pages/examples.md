title: Examples

[TOC]

## Generic example

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
as part of this distribution which contains an easily adaptable script to save your
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
! Import library for interfacing with PyTorch
use ftorch

implicit none

! Generate an object to hold the Torch model
type(torch_module) :: model

! Set up array of n_inputs input tensors and the output tensor
! Note: In this example there is only one input tensor (n_inputs = 1)
integer, parameter :: n_inputs = 1
type(torch_tensor), dimension(n_inputs) :: model_input_arr
type(torch_tensor) :: model_output

! Set up the model inputs and output as Fortran arrays
real, dimension(10,10), target  :: input
real, dimension(5), target   :: output

! Set up number of dimensions of input tensor and axis order
integer, parameter :: in_dims = 2
integer :: in_layout(in_dims) = [1,2]
integer, parameter :: out_dims = 1
integer :: out_layout(out_dims) = [1]

! Initialise the Torch model to be used
model = torch_module_load("/path/to/saved/model.pt")

! Initialise the inputs as Fortran array of ones
input = 1.0

! Wrap Fortran data as no-copy Torch Tensors
! There may well be some reshaping required depending on the 
! structure of the model which is not covered here (see examples)
model_input_arr(1) = torch_tensor_from_array(input, in_layout, torch_kCPU)
model_output = torch_tensor_from_array(output, out_layout, torch_kCPU)

! Run model and Infer
! Again, there may be some reshaping required depending on model design
call torch_module_forward(model, model_input_arr, n_inputs, model_output)

! Write out the result of running the model
write(*,*) output

! Clean up
call torch_module_delete(model)
call torch_tensor_delete(model_input_arr(1))
call torch_tensor_delete(model_output)
```

#### 3. Build the code

The code now needs to be compiled and linked against our installed library.
Here we describe how to do this for two build systems, CMake and make.

##### CMake
If our project were using CMake we would need the following in the `CMakeLists.txt`
file to find the FTorch installation and link it to the executable.

This can be done by adding the following to the `CMakeLists.txt` file:
```CMake
find_package(FTorch)
target_link_libraries( <executable> PRIVATE FTorch::ftorch )
message(STATUS "Building with Fortran PyTorch coupling")
```
and using the `-DCMAKE_PREFIX_PATH=</path/to/install/location>` flag when running CMake.  

> Note: _If you used the `CMAKE_INSTALL_PREFIX` argument when
> [building and installing the library](https://cambridge-iccs.github.io/FTorch/page/cmake.html)
> then you should use the same path for `</path/to/install/location>`._

##### Make
To build with make we need to include the library when compiling and link the executable
against it.

To compile with make we need add the following compiler flag when compiling files that
use FTorch:
```
FCFLAGS += -I<path/to/install/location>/include/ftorch
```

When compiling the final executable add the following link flag:
```
LDFLAGS += -L<path/to/install/location>/lib -lftorch
```

You may also need to add the location of the `.so` files to your `LD_LIBRARY_PATH`
unless installing in a default location:
```
export LD_LIBRARY_PATH = $LD_LIBRARY_PATH:<path/to/install/location>/lib
```
> Note: _Depending on your system and architecture `lib` may be `lib64` or something similar._

### Running on GPUs

In order to run a model on GPU, two main changes to the above process are required:

1. When saving your TorchScript model, ensure that it is on the GPU.
2. When calling `torch_tensor_from_array` in Fortran, the device for the input
   tensor(s) should be set to `torch_kCUDA`, rather than `torch_kCPU`.

For more information refer to the [GPU Documentation](gpu.html)

## Worked examples

The repository comes with a number of documented
[worked examples](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples).

These are designed to introduce users to FTorch and how to use the various features.

A subset of the examples are included as integration tests as part of FTorch's
[test suite](testing.html).

#### 1) SimpleNet

[This worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/1_SimpleNet)
provides a simple but complete demonstration of how to use the library.
It uses simple PyTorch 'net' that takes an input vector of length 5 and applies a single
Linear layer to multiply it by 2.
The aim is to demonstrate the most basic features of coupling before worrying about
more complex issues that are covered in later examples.

#### 2) Resnet

[This worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/2_ResNet18)
provides a more realistic demonstration of how to use the library,
using ResNet-18 to classify an image.
As the input to this model is four-dimensional (batch size, colour, x, y),
care must be taken dealing with the data array in Python and Fortran.
See [when to transpose arrays](transposing.html) for more details.

#### 3) MultiGPU

[This worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/3_MultiGPU)
builds on the SimpleNet demo and shows how to account for the case of sending different
data to multiple GPU devices.
