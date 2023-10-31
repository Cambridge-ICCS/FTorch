title: GPU Support

[TOC]

## GPU Support

In order to run a model on GPU, two main changes are required:

1) When saving your TorchScript model, ensure that it is on the GPU.
For example, when using [pt2ts.py](utils/pt2ts.py), this can be done by
uncommenting the following lines:  

``` { .python}
device = torch.device('cuda')
trained_model = trained_model.to(device)
trained_model.eval()
trained_model_dummy_input_1 = trained_model_dummy_input_1.to(device)
trained_model_dummy_input_2 = trained_model_dummy_input_2.to(device)
```
Note: this also moves the dummy input tensors to the GPU. This is not necessary for
saving the model, but the tensors must also be on the GPU to test that the models runs.

2) When calling `torch_tensor_from_blob` in Fortran, the device for the input tensor(s),
   but not the output tensor(s), should be set to `torch_kCUDA`, rather than
   `torch_kCPU`. This ensures that the inputs are on the same device as the model.

