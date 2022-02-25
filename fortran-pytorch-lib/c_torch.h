#ifndef C_TORCH_H
#define C_TORCH_H

#ifdef __cplusplus
#define EXPORT_C extern "C"
#else
#define EXPORT_C
#endif

// Opaque pointer type alias for torch::jit::script::Module class
typedef void *torch_jit_script_module_t;

// Opaque pointer type alias for at::Tensor
typedef void *torch_tensor_t;

// Data types
typedef enum
{
  torch_kUInt8,
  torch_kInt8,
  torch_kInt16,
  torch_kInt32,
  torch_kInt64,
  torch_kFloat16,
  torch_kFloat32,
  torch_kFloat64
} torch_data_t;

// Device types
typedef enum
{
  torch_kCPU,
  torch_kCUDA
} torch_device_t;

// Tensor API
EXPORT_C torch_tensor_t torch_zeros(const int ndim, const int64_t *shape,
                                    const torch_data_t dtype,
                                    const torch_device_t device);
EXPORT_C torch_tensor_t torch_ones(const int ndim, const int64_t *shape,
                                   const torch_data_t dtype,
                                   const torch_device_t device);
EXPORT_C void torch_ones2(const int ndim, const int64_t *shape,
                          const torch_data_t dtype, const torch_device_t device,
                          torch_tensor_t *out);
EXPORT_C torch_tensor_t torch_empty(const int ndim, const int64_t *shape,
                                    const torch_data_t dtype,
                                    const torch_device_t device);
EXPORT_C torch_tensor_t torch_from_blob(void *data, const int ndim,
                                        const int64_t *shape,
                                        const torch_data_t dtype,
                                        const torch_device_t device);
EXPORT_C void torch_tensor_print(torch_tensor_t tensor);
EXPORT_C void torch_tensor_delete(torch_tensor_t tensor);

// Module API
EXPORT_C torch_jit_script_module_t torch_jit_load(const char *filename);
EXPORT_C torch_tensor_t
torch_jit_module_forward(torch_jit_script_module_t module, torch_tensor_t input);
EXPORT_C void torch_jit_module_delete(torch_jit_script_module_t module);

// void torch_jit_optimize_for_inference();

#endif /* C_TORCH_H*/
