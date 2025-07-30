/*
 * For more details on the Torch Tensor C++ API, we refer to the Torch C++ documentation
 * (https://pytorch.org/cppdocs) and more specifically the C++ API documentation
 * (https://pytorch.org/cppdocs/api/library_root.html) pages on the PyTorch website.
 */
#include <torch/script.h>
#include <torch/torch.h>

#include "ctorch.h"

#ifndef GPU_DEVICE
#define GPU_DEVICE GPU_DEVICE_NONE
#endif

// =============================================================================
// --- Functions to aid in consistent error handling
// =============================================================================

// Accept a string message and handle as error. Accepts a cleanup function if desired.
void ctorch_error(const std::string &message,
                  const std::function<void()> &cleanup = nullptr) {
  std::cerr << "[ERROR]: " << message << std::endl;
  if (cleanup) {
    cleanup(); // Perform cleanup actions
  }
  exit(EXIT_FAILURE);
}

// Accept a string message and handle as a warning.
void ctorch_warn(const std::string &message) {
  std::cerr << "[WARNING]: " << message << std::endl;
}

// =============================================================================
// --- Constant expressions
// =============================================================================

// Mapping from FTorch device_data_t to libtorch Dtype
constexpr auto get_libtorch_dtype(torch_data_t dtype) {
  switch (dtype) {
  case torch_kUInt8:
    ctorch_error("uint8 not supported in Fortran");
    // See https://gcc.gnu.org/onlinedocs/gfortran/ISO_005fFORTRAN_005fENV.html
  case torch_kInt8:
    return torch::kInt8;
  case torch_kInt16:
    return torch::kInt16;
  case torch_kInt32:
    return torch::kInt32;
  case torch_kInt64:
    return torch::kInt64;
  case torch_kFloat16:
    ctorch_error("float16 not supported in Fortran");
    // See https://gcc.gnu.org/onlinedocs/gfortran/ISO_005fFORTRAN_005fENV.html
  case torch_kFloat32:
    return torch::kFloat32;
  case torch_kFloat64:
    return torch::kFloat64;
  default:
    ctorch_warn("unknown data type, setting to torch_kFloat32");
    return torch::kFloat32;
  }
}

// Mapping from libtorch Dtype to FTorch device_data_t
torch_data_t get_ftorch_dtype(caffe2::TypeMeta dtype) {
  if (dtype == torch::kUInt8) {
    ctorch_error("uint8 not supported in Fortran");
    // See https://gcc.gnu.org/onlinedocs/gfortran/ISO_005fFORTRAN_005fENV.html
  } else if (dtype == torch::kInt8) {
    return torch_kInt8;
  } else if (dtype == torch::kInt16) {
    return torch_kInt16;
  } else if (dtype == torch::kInt32) {
    return torch_kInt32;
  } else if (dtype == torch::kInt64) {
    return torch_kInt64;
  } else if (dtype == torch::kFloat16) {
    ctorch_error("float16 not supported in Fortran");
    // See https://gcc.gnu.org/onlinedocs/gfortran/ISO_005fFORTRAN_005fENV.html
  } else if (dtype == torch::kFloat32) {
    return torch_kFloat32;
  } else if (dtype == torch::kFloat64) {
    return torch_kFloat64;
  } else {
    std::cerr << "[ERROR]: data type " << dtype << " not supported in Fortran"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  return torch_kFloat32; // Dummy return to satisfy the compiler
}

// Mapping from FTorch device_type_t to libtorch DeviceType
const auto get_libtorch_device(torch_device_t device_type, int device_index) {
  switch (device_type) {
  case torch_kCPU:
    if (device_index != -1) {
      ctorch_warn("device index unused for CPU-only runs");
    }
    return torch::Device(torch::kCPU);
#if (GPU_DEVICE == GPU_DEVICE_CUDA) || (GPU_DEVICE == GPU_DEVICE_HIP)
  // NOTE: HIP is treated as CUDA in this project
  case torch_kCUDA:
    if (device_index == -1) {
      ctorch_warn("device index unset, defaulting to 0");
      device_index = 0;
    }
    if (device_index >= 0 && device_index < torch::cuda::device_count()) {
      return torch::Device(torch::kCUDA, device_index);
    } else {
      std::cerr << "[ERROR]: invalid device index " << device_index
                << " for device count " << torch::cuda::device_count() << std::endl;
      exit(EXIT_FAILURE);
    }
#endif
  case torch_kMPS:
    if (device_index != -1 && device_index != 0) {
      ctorch_warn("Only one device is available for MPS runs");
    }
    return torch::Device(torch::kMPS);
#if GPU_DEVICE == GPU_DEVICE_XPU
  case torch_kXPU:
    if (device_index == -1) {
      ctorch_warn("device index unset, defaulting to 0");
      device_index = 0;
    }
    if (device_index >= 0 && device_index < torch::xpu::device_count()) {
      return torch::Device(torch::kXPU, device_index);
    } else {
      std::cerr << "[ERROR]: invalid device index " << device_index
                << " for XPU device count " << torch::xpu::device_count() << std::endl;
      exit(EXIT_FAILURE);
    }
#endif
  default:
    ctorch_warn("unknown device type, setting to torch_kCPU");
    return torch::Device(torch::kCPU);
  }
}

// Mapping from libtorch DeviceType to FTorch device_type_t
const torch_device_t get_ftorch_device(torch::DeviceType device_type) {
  switch (device_type) {
  case torch::kCPU:
    return torch_kCPU;
  case torch::kCUDA:
    return torch_kCUDA;
  case torch::kHIP:
    return torch_kHIP;
  case torch::kXPU:
    return torch_kXPU;
  case torch::kMPS:
    return torch_kMPS;
  default:
    std::cerr << "[ERROR]: device type " << device_type << " not implemented in FTorch"
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

// =============================================================================
// --- Functions for validating tensors
// =============================================================================

// Check if a tensor is valid
void validate_tensor_not_null(const torch::Tensor *t, const std::string &name) {
  if (!t) {
    throw std::invalid_argument(name + " is null.");
  }
}

// Check if a tensor is defined
void validate_tensor_defined(const torch::Tensor *t, const std::string &name) {
  if (!t->defined()) {
    throw std::invalid_argument(name + " is undefined.");
  }
}

void validate_tensor(const torch::Tensor *t, const std::string &name) {
  validate_tensor_not_null(t, name);
  validate_tensor_defined(t, name);
}

// Check if a tensor has requires_grad set
void validate_requires_grad(const torch::Tensor *t, const std::string &name) {
  if (!t->requires_grad()) {
    throw std::runtime_error(name + " does not have requires_grad set.");
  }
}

void validate_gradient_defined(const torch::Tensor *t, const std::string &name) {
  if (!t->grad().defined()) {
    throw std::runtime_error(
        name + " has an undefined gradient.\nPerhaps you forgot to call backward.");
  }
}

// =============================================================================
// --- Functions for constructing tensors
// =============================================================================

torch_tensor_t torch_empty(int ndim, const int64_t *shape, torch_data_t dtype,
                           torch_device_t device_type, int device_index = -1,
                           const bool requires_grad = false) {
  torch::AutoGradMode enable_grad(requires_grad);
  auto tensor = new torch::Tensor;
  try {
    // This doesn't throw if shape and dimensions are incompatible
    c10::IntArrayRef vshape(shape, ndim);
    auto options = torch::TensorOptions()
                       .dtype(get_libtorch_dtype(dtype))
                       .device(get_libtorch_device(device_type, device_index))
                       .requires_grad(requires_grad);
    *tensor = torch::empty(vshape, options);
  } catch (const torch::Error &e) {
    ctorch_error(e.msg(), [&]() { delete tensor; });
  } catch (const std::exception &e) {
    ctorch_error(e.what(), [&]() { delete tensor; });
  }
  return tensor;
}

torch_tensor_t torch_zeros(int ndim, const int64_t *shape, torch_data_t dtype,
                           torch_device_t device_type, int device_index = -1,
                           const bool requires_grad = false) {
  torch::AutoGradMode enable_grad(requires_grad);
  auto tensor = new torch::Tensor;
  try {
    // This doesn't throw if shape and dimensions are incompatible
    c10::IntArrayRef vshape(shape, ndim);
    auto options = torch::TensorOptions()
                       .dtype(get_libtorch_dtype(dtype))
                       .device(get_libtorch_device(device_type, device_index))
                       .requires_grad(requires_grad);
    *tensor = torch::zeros(vshape, options);
  } catch (const torch::Error &e) {
    ctorch_error(e.msg(), [&]() { delete tensor; });
  } catch (const std::exception &e) {
    ctorch_error(e.what(), [&]() { delete tensor; });
  }
  return tensor;
}

torch_tensor_t torch_ones(int ndim, const int64_t *shape, torch_data_t dtype,
                          torch_device_t device_type, int device_index = -1,
                          const bool requires_grad = false) {
  torch::AutoGradMode enable_grad(requires_grad);
  auto tensor = new torch::Tensor;
  try {
    // This doesn't throw if shape and dimensions are incompatible
    c10::IntArrayRef vshape(shape, ndim);
    auto options = torch::TensorOptions()
                       .dtype(get_libtorch_dtype(dtype))
                       .device(get_libtorch_device(device_type, device_index))
                       .requires_grad(requires_grad);
    *tensor = torch::ones(vshape, options);
  } catch (const torch::Error &e) {
    ctorch_error(e.msg(), [&]() { delete tensor; });
  } catch (const std::exception &e) {
    ctorch_error(e.what(), [&]() { delete tensor; });
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
  auto tensor = new torch::Tensor;

  try {
    // This doesn't throw if shape and dimensions are incompatible
    c10::IntArrayRef vshape(shape, ndim);
    c10::IntArrayRef vstrides(strides, ndim);
    // NOTE: Do not pass device TensorOptions since this would cause torch::from_blob
    //       to expect data to reside on the host device_type, which is not case if
    //       device_type is a GPU device (see #365)
    auto options = torch::TensorOptions()
                       .dtype(get_libtorch_dtype(dtype))
                       .requires_grad(requires_grad);
    *tensor = torch::from_blob(data, vshape, vstrides, options)
                  .to(get_libtorch_device(device_type, device_index));

  } catch (const torch::Error &e) {
    ctorch_error(e.msg(), [&]() { delete tensor; });
  } catch (const std::exception &e) {
    ctorch_error(e.what(), [&]() { delete tensor; });
  }
  return tensor;
}

// =====================================================================================
// --- Functions for interrogating tensors
// =====================================================================================

void torch_tensor_print(const torch_tensor_t tensor) {
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  std::cout << *t << std::endl;
}

int torch_tensor_get_rank(const torch_tensor_t tensor) {
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  return t->sizes().size();
}

const torch_size_t *torch_tensor_get_sizes(const torch_tensor_t tensor) {
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  return t->sizes().data();
}

const torch_size_t *torch_tensor_get_stride(const torch_tensor_t tensor) {
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  return t->strides().data();
}

torch_data_t torch_tensor_get_dtype(const torch_tensor_t tensor) {
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  return get_ftorch_dtype(t->dtype());
}

torch_device_t torch_tensor_get_device_type(const torch_tensor_t tensor) {
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  return get_ftorch_device(t->device().type());
}

int torch_tensor_get_device_index(const torch_tensor_t tensor) {
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  return t->device().index();
}

bool torch_tensor_requires_grad(const torch_tensor_t tensor) {
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  return t->requires_grad();
}

// =====================================================================================
// --- Functions for deallocating tensors
// =====================================================================================

void torch_tensor_delete(torch_tensor_t tensor) {
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  delete t;
}

// =====================================================================================
// --- Functions for manipulating tensors
// =====================================================================================

void torch_tensor_zero(torch_tensor_t tensor) {
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  validate_tensor(t, "Input tensor");
  t->zero_();
}

void torch_tensor_to(const torch_tensor_t source_tensor, torch_tensor_t target_tensor,
                     bool non_blocking) {
  auto source_tens = reinterpret_cast<torch::Tensor *>(source_tensor);
  auto target_tens = reinterpret_cast<torch::Tensor *>(target_tensor);
  validate_tensor(source_tens, "Source tensor");
  validate_tensor(target_tens, "Target tensor");

  torch::Device device_type = target_tens->device();
  at::ScalarType dtype = target_tens->scalar_type();

  // For non-blocking usage see:
  // https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html
  std::move(*target_tens) = source_tens->to(device_type, dtype, non_blocking);
}

// =====================================================================================
// --- Operator overloads acting on tensors
// =====================================================================================

void torch_tensor_assign(torch_tensor_t output, const torch_tensor_t input) {
  auto out = reinterpret_cast<torch::Tensor *>(output);
  auto in = reinterpret_cast<torch::Tensor *const>(input);
  validate_tensor(out, "Output tensor");
  validate_tensor(in, "Input tensor");
  torch::AutoGradMode enable_grad(in->requires_grad());
  // NOTE: The following line ensures that the output tensor continues to point to a
  //       Fortran array if it was set up to do so using torch_tensor_from_array. If
  //       it's removed then the Fortran array keeps its original value and is no
  //       longer be pointed to.
  std::move(*out) = *in;
  // NOTE: The following line ensures that we always overwrite the requires_grad
  // property matching the PyTorch behaviour. See the Python examples on
  // https://github.com/Cambridge-ICCS/FTorch/pull/373.
  out->requires_grad_(in->requires_grad());
}

void torch_tensor_add(torch_tensor_t output, const torch_tensor_t tensor1,
                      const torch_tensor_t tensor2) {
  auto out = reinterpret_cast<torch::Tensor *>(output);
  auto t1 = reinterpret_cast<torch::Tensor *const>(tensor1);
  auto t2 = reinterpret_cast<torch::Tensor *const>(tensor2);
  *out = *t1 + *t2;
}

void torch_tensor_negative(torch_tensor_t output, const torch_tensor_t tensor) {
  auto out = reinterpret_cast<torch::Tensor *>(output);
  auto t = reinterpret_cast<torch::Tensor *const>(tensor);
  *out = -*t;
}

void torch_tensor_subtract(torch_tensor_t output, const torch_tensor_t tensor1,
                           const torch_tensor_t tensor2) {
  auto out = reinterpret_cast<torch::Tensor *>(output);
  auto t1 = reinterpret_cast<torch::Tensor *const>(tensor1);
  auto t2 = reinterpret_cast<torch::Tensor *const>(tensor2);
  *out = *t1 - *t2;
}

void torch_tensor_multiply(torch_tensor_t output, const torch_tensor_t tensor1,
                           const torch_tensor_t tensor2) {
  auto out = reinterpret_cast<torch::Tensor *>(output);
  auto t1 = reinterpret_cast<torch::Tensor *const>(tensor1);
  auto t2 = reinterpret_cast<torch::Tensor *const>(tensor2);
  *out = *t1 * *t2;
}

void torch_tensor_divide(torch_tensor_t output, const torch_tensor_t tensor1,
                         const torch_tensor_t tensor2) {
  auto out = reinterpret_cast<torch::Tensor *>(output);
  auto t1 = reinterpret_cast<torch::Tensor *const>(tensor1);
  auto t2 = reinterpret_cast<torch::Tensor *const>(tensor2);
  *out = *t1 / *t2;
}

void torch_tensor_power_int(torch_tensor_t output, const torch_tensor_t tensor,
                            const torch_int_t exponent) {
  // NOTE: The following cast will only work for integer exponents
  auto out = reinterpret_cast<torch::Tensor *>(output);
  auto t = reinterpret_cast<torch::Tensor *const>(tensor);
  auto exp = reinterpret_cast<int *const>(exponent);
  *out = pow(*t, *exp);
}

void torch_tensor_power_float(torch_tensor_t output, const torch_tensor_t tensor,
                              const torch_float_t exponent) {
  // NOTE: The following cast will only work for floating point exponents
  auto out = reinterpret_cast<torch::Tensor *>(output);
  auto t = reinterpret_cast<torch::Tensor *const>(tensor);
  auto exp = reinterpret_cast<float *const>(exponent);
  *out = pow(*t, *exp);
}

// ============================================================================
// --- Other operators for computations involving tensors
// ============================================================================

void torch_tensor_sum(torch_tensor_t output, const torch_tensor_t tensor) {
  auto out = reinterpret_cast<torch::Tensor *>(output);
  auto t = reinterpret_cast<torch::Tensor *const>(tensor);

  if (torch_tensor_get_rank(output) != 1) {
    std::stringstream errmsg;
    errmsg << "Invalid rank of output tensor for sum\nrank="
           << torch_tensor_get_rank(output) << " != 1";
    ctorch_error(errmsg.str());
  }
  if (torch_tensor_get_sizes(output)[0] != 1) {
    std::stringstream errmsg;
    errmsg << "Invalid shape of output tensor for sum\nshape=["
           << torch_tensor_get_sizes(output)[0] << "] != [1]";
    ctorch_error(errmsg.str());
  }
  std::move(*out) = t->sum();
}

void torch_tensor_mean(torch_tensor_t output, const torch_tensor_t tensor) {
  auto out = reinterpret_cast<torch::Tensor *>(output);
  auto t = reinterpret_cast<torch::Tensor *const>(tensor);

  if (torch_tensor_get_rank(output) != 1) {
    std::stringstream errmsg;
    std::cerr << "Invalid rank of output tensor for mean\nrank="
              << torch_tensor_get_rank(output) << " != 1";
    ctorch_error(errmsg.str());
  }
  if (torch_tensor_get_sizes(output)[0] != 1) {
    std::stringstream errmsg;
    errmsg << "Invalid shape of output tensor for mean\nshape=["
           << torch_tensor_get_sizes(output)[0] << "] != [1]";
    ctorch_error(errmsg.str());
  }
  std::move(*out) = t->mean();
}

// =============================================================================
// --- Functions related to automatic differentiation functionality for tensors
// =============================================================================

void torch_tensor_zero_grad(torch_tensor_t tensor) {
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  validate_tensor(t, "Gradient to zero");
  t->mutable_grad().zero_();
}

void torch_tensor_backward(const torch_tensor_t tensor,
                           const torch_tensor_t external_gradient,
                           const bool retain_graph) {
  auto t = reinterpret_cast<torch::Tensor *>(tensor);
  auto g = reinterpret_cast<torch::Tensor *const>(external_gradient);

  try {
    // Check if the tensors are valid and defined
    validate_tensor(t, "Input tensor");
    validate_tensor(g, "External gradient");

    // Perform backwards step
    t->backward(*g, retain_graph);
  } catch (const std::exception &e) {
    ctorch_error(std::string(e.what()) + " in torch_tensor_backward");
  }
}

void torch_tensor_get_gradient(const torch_tensor_t tensor, torch_tensor_t gradient) {
  try {
    // Cast the input pointers to torch::Tensor
    auto t = reinterpret_cast<torch::Tensor *const>(tensor);
    auto g = reinterpret_cast<torch::Tensor *>(gradient);

    // Check if the tensors are valid and defined
    validate_tensor(t, "Input tensor");
    validate_tensor_not_null(g, "Output gradient");
    // Check input has requires_grad set and can generate a valid gradient tensor
    validate_requires_grad(t, "Input tensor");
    validate_gradient_defined(t, "Input tensor");

    // Assign the gradient to the output tensor
    std::move(*g) = t->grad();
  } catch (const std::exception &e) {
    ctorch_error(std::string(e.what()) + " in torch_tensor_get_gradient");
  }
}

// =============================================================================
// --- Torch model API
// =============================================================================

void set_is_training(torch_jit_script_module_t module, const bool is_training = false) {
  auto model = static_cast<torch::jit::script::Module *>(module);
  if (is_training) {
    model->train();
  } else {
    model->eval();
  }
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
    *module =
        torch::jit::load(filename, get_libtorch_device(device_type, device_index));
  } catch (const torch::Error &e) {
    ctorch_error(e.msg(), [&]() { delete module; });
  } catch (const std::exception &e) {
    ctorch_error(e.what(), [&]() { delete module; });
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
      ctorch_error("One of the inputs to torch_jit_module_forward is not a Tensor");
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
      ctorch_error("Model Output is neither Tensor nor Tuple");
    }
  } catch (const torch::Error &e) {
    ctorch_error(e.msg());
  } catch (const std::exception &e) {
    ctorch_error(e.what());
  }
}

void torch_jit_module_delete(torch_jit_script_module_t module) {
  auto m = reinterpret_cast<torch::jit::script::Module *>(module);
  delete m;
}
