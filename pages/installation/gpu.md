title: GPU Support
author: Jack Atkinson
date: Last Updated: August 2026

## GPU Support

- [Dependencies](#dependencies)
- [Changes Required](#changes-required)
- [Multiple GPUs](#multiple-gpus)

FTorch supports running on a number of GPU hardwares by utilising the PyTorch/LibTorch
backends.
Currently supported are:

* CUDA (NVIDIA)
* HIP (AMD/ROCm)
* MPS (Apple Silicon)
* XPU (Intel)


### Dependencies

To run FTorch on different hardwares requires downloading the appropriate version of
Torch compatible with the device you wish to target.

This can be done for all hardwares by using a pip-installed version, and for CUDA and
HIP with a LibTorch binary.

#### Installation using pip

To install `ftorch_utils`, its dependencies, and the additional dependencies
for the examples with GPU support, use
```sh
pip install .[examples] --extra-index-url <pytorch-wheel-download-url>
```
or to install `torch` and `torchvision` directly use
```sh
pip install torch torchvision --index-url <pytorch-wheel-download-url>
```

In terms of the URL to use to download the appropriate wheel, instructions for
CPU, CUDA, and HIP/ROCm can be found in the installation matrix on
[pytorch.org](https://pytorch.org/). For XPU, use the
`https://download.pytorch.org/whl/test/xpu` wheel. For MPS, `pip` should
automatically detect the hardware and install the appropriate version so no
wheel needs to be specified.

#### LibTorch binary

For pure LibTorch binaries see the installation matrix on
[pytorch.org](https://pytorch.org/).
Currently standalone LibTorch binaries are only provided for CPU, CUDA, and HIP/ROCm.


### Changes Required

In order to run a model on GPU, three main changes are required:

**1) Build for the target device**  
When building FTorch, specify the target GPU architecture using the
[`GPU_DEVICE` CMake argument](|page|/installation/general.html#cmake-build-options):
```sh
cmake .. -DGPU_DEVICE=<CUDA/HIP/XPU/MPS>
```
The default setting is equivalent to
```sh
cmake .. -DGPU_DEVICE=NONE
```
i.e., CPU-only.

**2) Convert PyTorch model to TorchScript model**  
When converting a PyTorch model to a TorchScript model using the `pt2ts` script,
the device type should be specified via the `--device` option. For further
details on the `pt2ts` script, call `pt2ts --help` or read the
[ftorch-utils README](https://github.com/Cambridge-ICCS/FTorch/tree/main/ftorch_utils/README.md).

**3) Specify the target device from FTorch**  
When calling [[ftorch_tensor(module):torch_tensor_from_array(interface)]] and
[[ftorch_model(module):torch_model_load(subroutine)]] in Fortran, the device
for the input model should be set to the same device type that it was written
out using (`torch_kCUDA`, `torch_kHIP`, `torch_kXPU`, or `torch_kMPS`) rather
than `torch_kCPU`. Tensors should also be created on the appropriate device.

The following snippet shows how you would load a model to a CUDA device, create tensors,
and run inference:
```fortran
! Load in from Torchscript to device
call torch_model_load(torch_net, 'path/to/saved/model.pt', torch_kCUDA)

! Cast Fortran data to Tensors
call torch_tensor_from_array(input_tensors(1), in_data, torch_kCUDA)
call torch_tensor_from_array(output_tensors(1), out_data, torch_kCPU)

! Inference
call torch_model_forward(torch_net, input_tensors, output_tensors)
```

Note: _You do **not** need to change the device type for the output tensors as we
      want them to be on the CPU for subsequent use in Fortran._


### Multiple GPUs

For the case of having multiple GPU devices you should also specify a device index
of the GPU to be targeted for any input tensors and models in addition to
the device type. This argument is optional and will default to
device index 0 if unset.

For example, the following code snippet sets up a Torch tensor with CUDA GPU
device index 2:
```fortran
device_index = 2
call torch_tensor_from_array(in_tensors(1), in_data, tensor_layout, &
                             torch_kCUDA, device_index=device_index)
```
Whereas the following code snippet sets up a Torch tensor with (default) CUDA
device index 0:
```fortran
call torch_tensor_from_array(in_tensors(1), in_data, tensor_layout, &
                             torch_kCUDA)
```
Similarly for the `HIP` or `XPU` device type.
Note that `MPS` does not currently support multiple devices, so the
default device index should always be used.

See the
[MultiGPU example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/07_MultiGPU)
for a worked example of running with multiple devices from one code.
