title: GPU Support

[TOC]

## GPU Support

FTorch supports running on a number of GPU hardwares by utilising the PyTorch/LibTorch
backends.
Currently supported are:

* CUDA (NVIDIA)
* HIP (AMD/ROCm)
* MPS (Apple silicon)
* XPU (Intel)

> Note: _The HIP/ROCm backend uses the same API as the CUDA backend, so FTorch treats
> HIP as CUDA in places when calling LibTorch or PyTorch.
> This should not concern end-users as the FTorch and pt2ts.py APIs handle this.
> For further information see the
> [PyTorch HIP documentation](https://docs.pytorch.org/docs/stable/notes/hip.html)_


## Obtaining appropriate versions of PyTorch/LibTorch

To run FTorch on different hardwares requires downloading the appropriate version of
PyTorch/LibTorch.

This can be done for all hardwares by using a pip-installed version, and for CUDA and
HIP with a LibTorch binary.

### Using pip

If installing using pip the appropriate version for the hardware can be specified by
using the `--index-url` option during `pip install`.

Instructions for CPU, CUDA, and HIP/ROCm can be found in the installation matrix on
[pytorch.org](https://pytorch.org/).

For XPU use `--index-url https://download.pytorch.org/whl/test/xpu`, whilst for MPS
pip should automatically detect the hardware and install the appropriate version.

### LibTorch binary

For pure LibTorch binaries see the installation matrix on
[pytorch.org](https://pytorch.org/).
Currently standalone LibTorch binaries are only provided for CPU, CUDA, and HIP/ROCm.


## Changes required to run on GPU

In order to run a model on GPU, three main changes are required:

1) When building FTorch, specify the target GPU architecture using the
`GPU_DEVICE` argument. That is, set
```sh
cmake .. -DGPU_DEVICE=<CUDA/HIP/XPU/MPS>
```
as appropriate. The default setting is equivalent to
```sh
cmake .. -DGPU_DEVICE=NONE
```
i.e., CPU-only.

2) When saving your TorchScript model, ensure that it is on the GPU.
For example, when using
[`pt2ts.py`](https://github.com/Cambridge-ICCS/FTorch/blob/main/utils/pt2ts.py),
this can be done by passing the `--device_type <cuda/hip/xpu/mps>` argument. This
sets the `device_type` variable, which has the effect of transferring the model
and any input arrays to the specified GPU device in the following lines:
```python
if device_type != "cpu":
    trained_model = trained_model.to(device_type)
    trained_model.eval()
    trained_model_dummy_input_1 = trained_model_dummy_input_1.to(device_type)
    trained_model_dummy_input_2 = trained_model_dummy_input_2.to(device_type)
```

> Note: _This code moves the dummy input tensors to the GPU, as well as the
> model.
> Whilst this is not necessary for saving the model the tensors must be on
> the same GPU device to test that the models runs._

3) When calling `torch_tensor_from_array` in Fortran, the device type for the
   input tensor(s) should be set to the relevant device type (`torch_kCUDA`,
   `torch_kHIP`, `torch_kXPU`, or `torch_kMPS`) rather than `torch_kCPU`.
   This ensures that the inputs are on the same device type as the model.

> Note: _You do **not** need to change the device type for the output tensors as we
> want them to be on the CPU for subsequent use in Fortran._

### Multi-GPU runs

In the case of having multiple GPU devices, as well as setting the device type
for any input tensors and models, you should also specify their device index
as the GPU device to be targeted. This argument is optional and will default to
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
Similarly for the XPU device type.

> Note: The MPS device type does not currently support multiple devices, so the
> default device index should always be used.

See the
[MultiGPU example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/6_MultiGPU)
for a worked example of running with multiple GPUs.
