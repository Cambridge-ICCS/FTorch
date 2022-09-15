#ifndef C_TORCH_H
#define C_TORCH_H

#ifdef __cplusplus
#define EXPORT_C extern "C"
#else
#define EXPORT_C
#endif

// Opaque pointer type alias for torch::jit::script::Module class
typedef void* torch_jit_script_module_t;

// Opaque pointer type alias for at::Tensor
typedef void* torch_tensor_t;

// Data types
typedef enum {
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
typedef enum { torch_kCPU, torch_kCUDA } torch_device_t;

// Tensor API
EXPORT_C torch_tensor_t torch_zeros(int ndim, const int64_t* shape,
                                    torch_data_t dtype, torch_device_t device);
EXPORT_C torch_tensor_t torch_ones(int ndim, const int64_t* shape,
                                   torch_data_t dtype, torch_device_t device);
EXPORT_C torch_tensor_t torch_empty(int ndim, const int64_t* shape,
                                    torch_data_t dtype, torch_device_t device);
EXPORT_C torch_tensor_t torch_from_blob(void* data, int ndim,
                                        const int64_t* shape,
                                        torch_data_t dtype,
                                        torch_device_t device);
EXPORT_C void torch_tensor_print(const torch_tensor_t tensor);
EXPORT_C void torch_tensor_delete(torch_tensor_t tensor);

// Module API
EXPORT_C torch_jit_script_module_t torch_jit_load(const char* filename);
EXPORT_C void torch_jit_module_forward(const torch_jit_script_module_t module,
                                       const torch_tensor_t input,
                                       torch_tensor_t output);
EXPORT_C void torch_jit_module_delete(torch_jit_script_module_t module);

#endif /* C_TORCH_H*/
