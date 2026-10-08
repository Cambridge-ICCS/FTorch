title: API Equivalents
author: FTorch contributors
date: Last Updated: October 2026

## PyTorch, libtorch, and FTorch API Equivalents

This page maps common types and operations across the PyTorch Python, libtorch
C++, and FTorch Fortran APIs. It is a quick reference, not a complete API
listing: the interfaces are not always one-to-one, and FTorch exposes the
PyTorch features needed for calling TorchScript models from Fortran.

### Types

| Concept | PyTorch (Python) | libtorch (C++) | FTorch (Fortran) |
| ----------------- | ------------------------------ | ------------------------------ | ------------------------------ |
| Tensor | [`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [`torch::Tensor`](https://docs.pytorch.org/cppdocs/api/aten/tensor.html) | [[ftorch_tensor(module):torch_tensor(type)]] |
| TorchScript model | [`torch.jit.ScriptModule`](https://docs.pytorch.org/docs/stable/jit.html#torch.jit.ScriptModule) | [`torch::jit::Module`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/api/module.h) | [[ftorch_model(module):torch_model(type)]] |
| Optimizer | [`torch.optim.Optimizer`](https://docs.pytorch.org/docs/stable/optim.html#torch.optim.Optimizer) | [`torch::optim::Optimizer`](https://docs.pytorch.org/cppdocs/api/optim/index.html#optimizer-base-class) | [[ftorch_optim(module):torch_optim(type)]] |

### Tensors

| Operation | PyTorch (Python) | libtorch (C++) | FTorch (Fortran) |
| ------------------------ | ------------------------------ | ------------------------------ | ------------------------------ |
| Create a tensor of zeros | [`torch.zeros`](https://docs.pytorch.org/docs/stable/generated/torch.zeros.html) | [`torch::zeros`](https://docs.pytorch.org/cppdocs/api/aten/creation.html) | [[ftorch_tensor(module):torch_tensor_zeros(subroutine)]] |
| Get shape | [`Tensor.size`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.size.html) | [`Tensor::sizes`](https://docs.pytorch.org/cppdocs/api/aten/tensor.html) | [[ftorch_tensor(module):torch_tensor_get_shape(function)]] |
| Get strides | [`Tensor.stride`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.stride.html) | [`Tensor::strides`](https://docs.pytorch.org/cppdocs/api/aten/tensor.html) | [[ftorch_tensor(module):torch_tensor_get_stride(function)]] |
| Get data type | [`Tensor.dtype`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [`Tensor::scalar_type`](https://docs.pytorch.org/cppdocs/api/aten/tensor.html) | [[ftorch_tensor(module):torch_tensor_get_dtype(function)]] |
| Get device | [`Tensor.device`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [`Tensor::device`](https://docs.pytorch.org/cppdocs/api/aten/tensor.html) | [[ftorch_tensor(module):torch_tensor_get_device_type(function)]] and [[ftorch_tensor(module):torch_tensor_get_device_index(function)]] |
| Backpropagate gradients | [`Tensor.backward`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html) | [`torch::autograd::backward`](https://docs.pytorch.org/cppdocs/api/autograd/gradient.html) | [[ftorch_tensor(module):torch_tensor_backward(interface)]] |

### Models

| Operation | PyTorch (Python) | libtorch (C++) | FTorch (Fortran) |
| ------------------------ | ------------------------------ | ------------------------------ | ------------------------------ |
| Load a TorchScript model | [`torch.jit.load`](https://docs.pytorch.org/docs/stable/jit.html#torch.jit.load) | [`torch::jit::load`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/serialization/import.h) | [[ftorch_model(module):torch_model_load(subroutine)]] |
| Run a forward pass | [`Module.forward`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.forward) | [`torch::jit::Module::forward`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/api/module.h) | [[ftorch_model(module):torch_model_forward(subroutine)]] |

### Optimizers

| Operation | PyTorch (Python) | libtorch (C++) | FTorch (Fortran) |
| -------------------- | ------------------------------ | ------------------------------ | ------------------------------ |
| Create SGD optimizer | [`torch.optim.SGD`](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html) | [`torch::optim::SGD`](https://docs.pytorch.org/cppdocs/api/optim/gradient_descent.html) | [[ftorch_optim(module):torch_optim_SGD(subroutine)]] |
| Clear gradients | [`Optimizer.zero_grad`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html) | [`Optimizer::zero_grad`](https://docs.pytorch.org/cppdocs/api/optim/index.html#optimizer-base-class) | [[ftorch_optim(module):torch_optim_zero_grad(subroutine)]] (`optimizer%zero_grad()`) |
| Update parameters | [`Optimizer.step`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html) | [`Optimizer::step`](https://docs.pytorch.org/cppdocs/api/optim/index.html#optimizer-base-class) | [[ftorch_optim(module):torch_optim_step(subroutine)]] (`optimizer%step()`) |

The model entries refer to TorchScript. Although TorchScript is deprecated in
current PyTorch documentation, FTorch's model API currently uses TorchScript
models.

For details and usage examples, see the [Tensor API](|page|/usage/tensor.html),
[Optimizers API](|page|/usage/optimizers.html), and
[online training](|page|/usage/online.html) pages.
