/*
 * For more details on the Torch Tensor C++ API, we refer to the Torch C++ documentation
 * (https://pytorch.org/cppdocs) and more specifically the C++ API documentation
 * (https://pytorch.org/cppdocs/api/library_root.html) pages on the PyTorch website.
 */
#include <torch/script.h>
#include <torch/torch.h>

#include "ctorch.h"

constexpr auto get_dtype(torch_data_t dtype) {
  switch (dtype) {
  case torch_kUInt8:
    std::cerr << "[WARNING]: uint8 not supported in Fortran" << std::endl;
    // See https://gcc.gnu.org/onlinedocs/gfortran/ISO_005fFORTRAN_005fENV.html
    exit(EXIT_FAILURE);
  case torch_kInt8:
    return torch::kInt8;
  case torch_kInt16:
    return torch::kInt16;
  case torch_kInt32:
    return torch::kInt32;
  case torch_kInt64:
    return torch::kInt64;
  case torch_kFloat16:
    std::cerr << "[WARNING]: float16 not supported in Fortran" << std::endl;
    // See https://gcc.gnu.org/onlinedocs/gfortran/ISO_005fFORTRAN_005fENV.html
    exit(EXIT_FAILURE);
  case torch_kFloat32:
    return torch::kFloat32;
  case torch_kFloat64:
    return torch::kFloat64;
  default:
    std::cerr << "[WARNING]: unknown data type, setting to torch_kFloat32" << std::endl;
    return torch::kFloat32;
  }
}

const auto get_device(torch_device_t device_type, int device_index) {
  switch (device_type) {
  case torch_kCPU:
    if (device_index != -1) {
      std::cerr << "[WARNING]: device index unused for CPU-only runs" << std::endl;
    }
    return torch::Device(torch::kCPU);
  case torch_kCUDA:
    if (device_index == -1) {
      std::cerr << "[WARNING]: device index unset, defaulting to 0" << std::endl;
      device_index = 0;
    }
    if (device_index >= 0 && device_index < torch::cuda::device_count()) {
      return torch::Device(torch::kCUDA, device_index);
    } else {
      std::cerr << "[ERROR]: invalid device index " << device_index
                << " for device count " << torch::cuda::device_count() << std::endl;
      exit(EXIT_FAILURE);
    }
  default:
    std::cerr << "[WARNING]: unknown device type, setting to torch_kCPU" << std::endl;
    return torch::Device(torch::kCPU);
  }
}

void set_is_training(torch_jit_script_module_t module, const bool is_training = false) {
  auto model = static_cast<torch::jit::script::Module *>(module);
  if (is_training) {
    model->train();
  } else {
    model->eval();
  }
}

torch_tensor_t torch_zeros(int ndim, const int64_t *shape, torch_data_t dtype,
                           torch_device_t device_type, int device_index = -1,
                           const bool requires_grad = false) {
  torch::AutoGradMode enable_grad(requires_grad);
  torch::Tensor *tensor = nullptr;
  try {
    // This doesn't throw if shape and dimensions are incompatible
    c10::IntArrayRef vshape(shape, ndim);
    tensor = new torch::Tensor;
    *tensor = torch::zeros(vshape, torch::dtype(get_dtype(dtype)))
                  .to(get_device(device_type, device_index));
  } catch (const torch::Error &e) {
    std::cerr << "[ERROR]: " << e.msg() << std::endl;
    delete tensor;
    exit(EXIT_FAILURE);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR]: " << e.what() << std::endl;
    delete tensor;
    exit(EXIT_FAILURE);
  }
  return tensor;
}

torch_tensor_t torch_ones(int ndim, const int64_t *shape, torch_data_t dtype,
                          torch_device_t device_type, int device_index = -1,
                          const bool requires_grad = false) {
  torch::AutoGradMode enable_grad(requires_grad);
  torch::Tensor *tensor = nullptr;
  try {
    // This doesn't throw if shape and dimensions are incompatible
    c10::IntArrayRef vshape(shape, ndim);
    tensor = new torch::Tensor;
    *tensor = torch::ones(vshape, torch::dtype(get_dtype(dtype)))
                  .to(get_device(device_type, device_index));
  } catch (const torch::Error &e) {
    std::cerr << "[ERROR]: " << e.msg() << std::endl;
    delete tensor;
    exit(EXIT_FAILURE);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR]: " << e.what() << std::endl;
    delete tensor;
    exit(EXIT_FAILURE);
  }
  return tensor;
}

torch_tensor_t torch_empty(int ndim, const int64_t *shape, torch_data_t dtype,
                           torch_device_t device_type, int device_index = -1,
                           const bool requires_grad = false) {
  torch::AutoGradMode enable_grad(requires_grad);
  torch::Tensor *tensor = nullptr;
  try {
    // This doesn't throw if shape and dimensions are incompatible
    c10::IntArrayRef vshape(shape, ndim);
    tensor = new torch::Tensor;
    *tensor = torch::empty(vshape, torch::dtype(get_dtype(dtype)))
                  .to(get_device(device_type, device_index));
  } catch (const torch::Error &e) {
    std::cerr << "[ERROR]: " << e.msg() << std::endl;
    delete tensor;
    exit(EXIT_FAILURE);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR]: " << e.what() << std::endl;
    delete tensor;
    exit(EXIT_FAILURE);
  }
  return tensor;
}

// Exposes the given data as a Tensor without taking ownership of the original
// data
torch_tensor_t torch_from_blob(void *data, int ndim, const int64_t *shape,
                               const int64_t *strides, torch_data_t dtype,
                               torch_device_t device_type, int device_index = -1,
                               const bool requires_grad = false) {
  torch::AutoGradMode enable_grad(requires_grad);
  torch::Tensor *tensor = nullptr;

  try {
    // This doesn't throw if shape and dimensions are incompatible
    c10::IntArrayRef vshape(shape, ndim);
    c10::IntArrayRef vstrides(strides, ndim);
    tensor = new torch::Tensor;
    *tensor = torch::from_blob(data, vshape, vstrides, torch::dtype(get_dtype(dtype)))
                  .to(get_device(device_type, device_index));

  } catch (const torch::Error &e) {
    std::cerr << "[ERROR]: " << e.msg() << std::endl;
    delete tensor;
    exit(EXIT_FAILURE);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR]: " << e.what() << std::endl;
    delete tensor;
    exit(EXIT_FAILURE);
  }
  return tensor;
}

void *torch_to_blob(const torch_tensor_t tensor, const torch_data_t dtype) {
  auto t = reinterpret_cast<torch::Tensor *const>(tensor);
  void *raw_ptr;
  switch (dtype) {
  case torch_kUInt8:
    std::cerr << "[WARNING]: uint8 not supported" << std::endl;
    exit(EXIT_FAILURE);
  case torch_kInt8:
    raw_ptr = (void *)t->data_ptr<int8_t>();
    break;
  case torch_kInt16:
    raw_ptr = (void *)t->data_ptr<int16_t>();
    break;
  case torch_kInt32:
    raw_ptr = (void *)t->data_ptr<int32_t>();
    break;
  case torch_kInt64:
    raw_ptr = (void *)t->data_ptr<int64_t>();
    break;
  case torch_kFloat16:
    std::cerr << "[WARNING]: float16 not supported" << std::endl;
    // NOTE: std::float16_t is available but only with C++23
    exit(EXIT_FAILURE);
  case torch_kFloat32:
    raw_ptr = (void *)t->data_ptr<float>();
    // NOTE: std::float32_t is available but only with C++23
    break;
  case torch_kFloat64:
    raw_ptr = (void *)t->data_ptr<double>();
    // NOTE: std::float64_t is available but only with C++23
    break;
  default:
    std::cerr << "[WARNING]: unknown data type" << std::endl;
    exit(EXIT_FAILURE);
  }
  return raw_ptr;
}

void torch_tensor_print(const torch_tensor_t tensor) {
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  std::cout << *t << std::endl;
}

int torch_tensor_get_device_index(const torch_tensor_t tensor) {
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  return t->device().index();
}

int torch_tensor_get_rank(const torch_tensor_t tensor) {
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  return t->sizes().size();
}

#ifdef UNIX
const long int *torch_tensor_get_sizes(const torch_tensor_t tensor) {
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  return t->sizes().data();
}
#else
const long long int *torch_tensor_get_sizes(const torch_tensor_t tensor) {
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  return t->sizes().data();
}
#endif

void torch_tensor_delete(torch_tensor_t tensor) {
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  delete t;
}

torch_tensor_t torch_tensor_assign(const torch_tensor_t input) {
  auto in = reinterpret_cast<torch::Tensor *const>(input);
  torch::AutoGradMode enable_grad(in->requires_grad());
  torch::Tensor *output = nullptr;
  output = new torch::Tensor;
  *output = in->detach().clone();
  return output;
}

torch_tensor_t torch_tensor_add(const torch_tensor_t tensor1,
                                const torch_tensor_t tensor2) {
  auto t1 = reinterpret_cast<torch::Tensor *const>(tensor1);
  auto t2 = reinterpret_cast<torch::Tensor *const>(tensor2);
  torch::Tensor *output = nullptr;
  output = new torch::Tensor;
  *output = *t1 + *t2;
  return output;
}

torch_tensor_t torch_tensor_subtract(const torch_tensor_t tensor1,
                                     const torch_tensor_t tensor2) {
  auto t1 = reinterpret_cast<torch::Tensor *const>(tensor1);
  auto t2 = reinterpret_cast<torch::Tensor *const>(tensor2);
  torch::Tensor *output = nullptr;
  output = new torch::Tensor;
  *output = *t1 - *t2;
  return output;
}

torch_tensor_t torch_tensor_multiply(const torch_tensor_t tensor1,
                                     const torch_tensor_t tensor2) {
  auto t1 = reinterpret_cast<torch::Tensor *const>(tensor1);
  auto t2 = reinterpret_cast<torch::Tensor *const>(tensor2);
  torch::Tensor *output = nullptr;
  output = new torch::Tensor;
  *output = *t1 * *t2;
  return output;
}

torch_tensor_t torch_tensor_divide(const torch_tensor_t tensor1,
                                   const torch_tensor_t tensor2) {
  auto t1 = reinterpret_cast<torch::Tensor *const>(tensor1);
  auto t2 = reinterpret_cast<torch::Tensor *const>(tensor2);
  torch::Tensor *output = nullptr;
  output = new torch::Tensor;
  *output = *t1 / *t2;
  return output;
}

torch_tensor_t torch_tensor_power_int(const torch_tensor_t tensor,
                                      const torch_int_t exponent) {
  auto t = reinterpret_cast<torch::Tensor *const>(tensor);
  // NOTE: The following cast will only work for integer exponents
  auto exp = reinterpret_cast<int *const>(exponent);
  torch::Tensor *output = nullptr;
  output = new torch::Tensor;
  *output = pow(*t, *exp);
  return output;
}

torch_tensor_t torch_tensor_power_float(const torch_tensor_t tensor,
                                        const torch_float_t exponent) {
  auto t = reinterpret_cast<torch::Tensor *const>(tensor);
  // NOTE: The following cast will only work for floating point exponents
  auto exp = reinterpret_cast<float *const>(exponent);
  torch::Tensor *output = nullptr;
  output = new torch::Tensor;
  *output = pow(*t, *exp);
  return output;
}

torch_jit_script_module_t torch_jit_load(const char *filename,
                                         const torch_device_t device_type = torch_kCPU,
                                         const int device_index = -1,
                                         const bool requires_grad = false,
                                         const bool is_training = false) {
  torch::AutoGradMode enable_grad(requires_grad);
  torch::jit::script::Module *module = nullptr;
  try {
    module = new torch::jit::script::Module;
    *module = torch::jit::load(filename, get_device(device_type, device_index));
  } catch (const torch::Error &e) {
    std::cerr << "[ERROR]: " << e.msg() << std::endl;
    delete module;
    exit(EXIT_FAILURE);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR]: " << e.what() << std::endl;
    delete module;
    exit(EXIT_FAILURE);
  }
  set_is_training(module, is_training);

  return module;
}

void torch_jit_module_forward(const torch_jit_script_module_t module,
                              const torch_tensor_t *inputs, const int nin,
                              torch_tensor_t *outputs, const int nout,
                              const bool requires_grad = false) {
  torch::AutoGradMode enable_grad(requires_grad);
  // Here we cast the pointers we recieved in to Tensor objects
  auto model = static_cast<torch::jit::script::Module *>(module);
  auto in = reinterpret_cast<torch::Tensor *const *>(inputs);
  auto out = reinterpret_cast<torch::Tensor **>(outputs);
  // Local IValue for checking we are passed types
  torch::jit::IValue LocalTensor;
  // Generate a vector of IValues (placeholders for various Torch types)
  std::vector<torch::jit::IValue> inputs_vec;
  // Populate with Tensors pointed at by pointers
  // For each IValue check it is of Tensor type
  for (int i = 0; i < nin; ++i) {
    LocalTensor = *(in[i]);
    if (LocalTensor.isTensor()) {
      inputs_vec.push_back(LocalTensor);
    } else {
      std::cerr << "[ERROR]: One of the inputs to torch_jit_module_forward is "
                   "not a Tensor."
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  try {
    auto model_out = model->forward(inputs_vec);
    if (model_out.isTensor()) {
      // Single output models will return a tensor directly.
      std::move(*out[0]) = model_out.toTensor();
    } else if (model_out.isTuple()) {
      // Multiple output models will return a tuple => cast to tensors.
      for (int i = 0; i < nout; ++i) {
        std::move(*out[i]) = model_out.toTuple()->elements()[i].toTensor();
      }
    } else {
      // If for some reason the forward method does not return a Tensor it
      // should raise an error when trying to cast to a Tensor type
      std::cerr << "[ERROR]: Model Output is neither Tensor nor Tuple." << std::endl;
    }
  } catch (const torch::Error &e) {
    std::cerr << "[ERROR]: " << e.msg() << std::endl;
    exit(EXIT_FAILURE);
  } catch (const std::exception &e) {
    std::cerr << "[ERROR]: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }
}

void torch_jit_module_delete(torch_jit_script_module_t module) {
  auto m = reinterpret_cast<torch::jit::script::Module *>(module);
  delete m;
}
