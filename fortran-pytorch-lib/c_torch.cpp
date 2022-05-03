#include <torch/script.h>
#include <torch/torch.h>

#include "c_torch.h"

constexpr auto get_dtype(torch_data_t dtype)
{
  switch (dtype)
  {
  case torch_kUInt8:
    return torch::kUInt8;
  case torch_kInt8:
    return torch::kInt8;
  case torch_kInt16:
    return torch::kInt16;
  case torch_kInt32:
    return torch::kInt32;
  case torch_kInt64:
    return torch::kInt64;
  case torch_kFloat16:
    return torch::kFloat16;
  case torch_kFloat32:
    return torch::kFloat32;
  case torch_kFloat64:
    return torch::kFloat64;
  }
}

constexpr auto get_device(torch_device_t device)
{
  switch (device)
  {
  case torch_kCPU:
    return torch::kCPU;
  case torch_kCUDA:
    return torch::kCUDA;
  }
}

torch_tensor_t torch_zeros(const int ndim, const int64_t *shape,
                           const torch_data_t dtype,
                           const torch_device_t device)
{
  // FIXME: sanity checks
  c10::IntArrayRef vshape(shape, ndim);
  torch::Tensor *tensor = new torch::Tensor;
  *tensor = torch::zeros(
      vshape, torch::dtype(get_dtype(dtype)).device(get_device(device)));
  return tensor;
}

torch_tensor_t torch_ones(const int ndim, const int64_t *shape,
                          const torch_data_t dtype, const torch_device_t device)
{
  // FIXME: sanity checks
  c10::IntArrayRef vshape(shape, ndim);
  torch::Tensor *tensor = new torch::Tensor;
  *tensor = torch::ones(
      vshape, torch::dtype(get_dtype(dtype)).device(get_device(device)));
  return tensor;
}

torch_tensor_t torch_empty(const int ndim, const int64_t *shape,
                           const torch_data_t dtype,
                           const torch_device_t device)
{
  // FIXME: sanity checks
  c10::IntArrayRef vshape(shape, ndim);
  torch::Tensor *tensor = new torch::Tensor;
  *tensor = torch::empty(
      vshape, torch::dtype(get_dtype(dtype)).device(get_device(device)));
  return tensor;
}

// Exposes the given data as a Tensor without taking ownership of the original
// data
torch_tensor_t torch_from_blob(void *data, const int ndim, const int64_t *shape,
                               const torch_data_t dtype,
                               const torch_device_t device)
{
  // FIXME: sanity checks
  // Will the ctor throw an error
  c10::IntArrayRef vshape(shape, ndim);
  torch::Tensor *tensor = new torch::Tensor;
  *tensor = torch::from_blob(
      data, vshape, torch::dtype(get_dtype(dtype)).device(get_device(device)));
  return tensor;
}

void torch_tensor_print(torch_tensor_t tensor)
{
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  std::cout << *t << std::endl;
}

void torch_tensor_delete(torch_tensor_t tensor)
{
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  delete t;
}

torch_jit_script_module_t torch_jit_load(const char *filename)
{
  torch::jit::script::Module *module = new torch::jit::script::Module;
  try
  {
    *module = torch::jit::load(filename);
  }
  catch (const c10::Error &e)
  {
    std::cerr << "Error in loading the model" << e.msg() << std::endl;
    delete module;
    return nullptr;
  }

  return module;
}

torch_tensor_t torch_jit_module_forward(torch_jit_script_module_t module,
                                        torch_tensor_t input)
{
  // FIXME: Sanity checks
  auto mod = static_cast<torch::jit::script::Module *>(module);
  auto in = static_cast<torch::Tensor *>(input);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(*in);
  torch::Tensor *output = new torch::Tensor;
  *output = mod->forward(inputs).toTensor();
  if (output->is_cuda())
    // It uses the current device, given by current_device(), if device is None
    torch::cuda::synchronize();
  std::cout << output->slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl;
  return output;
}

void torch_jit_module_delete(torch_jit_script_module_t module)
{
  auto m = reinterpret_cast<torch::jit::script::Module *>(module);
  delete m;
}
