title: GPU Support

[TOC]

## GPU Support

In order to run a model on GPU, three main changes are required:

1) When building FTorch, specify the target GPU architecture using the
`GPU_DEVICE` argument. That is, set
```sh
cmake .. -DGPU_DEVICE=<CUDA/XPU/MPS>
```
as appropriate. The default setting is equivalent to
```sh
cmake .. -DGPU_DEVICE=NONE
```
i.e., CPU-only.

2) When saving your TorchScript model, ensure that it is on the GPU.
For example, when using
[`pt2ts.py`](https://github.com/Cambridge-ICCS/FTorch/blob/main/utils/pt2ts.py),
this can be done by passing the `--device_type <cuda/xpu/mps>` argument. This
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
   `torch_kXPU`, or `torch_kMPS`) rather than `torch_kCPU`.
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
[MultiGPU example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/3_MultiGPU)
for a worked example of running with multiple GPUs.
