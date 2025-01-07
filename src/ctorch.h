#ifndef C_TORCH_H
#define C_TORCH_H

#ifdef __cplusplus
#define EXPORT_C extern "C"
#else
#define EXPORT_C
#endif

#include <stdint.h>

// Opaque pointer type alias for torch::jit::script::Module class
typedef void *torch_jit_script_module_t;

// Opaque pointer type alias for at::Tensor
typedef void *torch_tensor_t;

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

// =====================================================================================
// Tensor API
// =====================================================================================
/**
 * Function to generate a Torch Tensor of zeros
 * @param number of dimensions of the Tensor
 * @param shape of the Tensor
 * @param data type of the elements of the Tensor
 * @param device type used (cpu, CUDA, etc.)
 * @param device index for the CUDA case
 * @param whether gradient is required
 */
EXPORT_C torch_tensor_t torch_zeros(int ndim, const int64_t *shape, torch_data_t dtype,
                                    torch_device_t device_type, int device_index,
                                    const bool requires_grad);

/**
 * Function to generate a Torch Tensor of ones
 * @param number of dimensions of the Tensor
 * @param shape of the Tensor
 * @param data type of the elements of the Tensor
 * @param device type used (cpu, CUDA, etc.)
 * @param device index for the CUDA case
 * @param whether gradient is required
 */
EXPORT_C torch_tensor_t torch_ones(int ndim, const int64_t *shape, torch_data_t dtype,
                                   torch_device_t device_type, int device_index,
                                   const bool requires_grad);

/**
 * Function to generate an empty Torch Tensor
 * @param number of dimensions of the Tensor
 * @param shape of the Tensor
 * @param data type of the elements of the Tensor
 * @param device type used (cpu, CUDA, etc.)
 * @param device index for the CUDA case
 * @param whether gradient is required
 */
EXPORT_C torch_tensor_t torch_empty(int ndim, const int64_t *shape, torch_data_t dtype,
                                    torch_device_t device_type, int device_index,
                                    const bool requires_grad);

/**
 * Function to create a Torch Tensor from memory location given extra
 * information
 * @param pointer to the Tensor in memory
 * @param number of dimensions of the Tensor
 * @param shape of the Tensor
 * @param strides to take through data
 * @param data type of the elements of the Tensor
 * @param device type used (cpu, CUDA, etc.)
 * @param device index for the CUDA case
 * @param whether gradient is required
 * @return Torch Tensor interpretation of the data pointed at
 */
EXPORT_C torch_tensor_t torch_from_blob(void *data, int ndim, const int64_t *shape,
                                        const int64_t *strides, torch_data_t dtype,
                                        torch_device_t device_type, int device_index,
                                        const bool requires_grad);

/**
 * Function to extract a C-array from a Torch Tensor's data.
 *
 * @param the Torch Tensor
 * @param data type of the elements of the Tensor
 * @return pointer to the Tensor in memory
 */
EXPORT_C void *torch_to_blob(const torch_tensor_t tensor, const torch_data_t dtype);

/**
 * Function to print out a Torch Tensor
 * @param Torch Tensor to print
 */
EXPORT_C void torch_tensor_print(const torch_tensor_t tensor);

/**
 * Function to determine the device index of a Torch Tensor
 * @param Torch Tensor to determine the device index of
 * @return device index of the Torch Tensor
 */
EXPORT_C int torch_tensor_get_device_index(const torch_tensor_t tensor);

/**
 * Function to determine the rank of a Torch Tensor
 * @param Torch Tensor to determine the rank of
 * @return rank of the Torch Tensor
 */
EXPORT_C int torch_tensor_get_rank(const torch_tensor_t tensor);

/**
 * Function to determine the sizes (shape) of a Torch Tensor
 * @param Torch Tensor to determine the rank of
 * @return pointer to the sizes array of the Torch Tensor
 */
#ifdef UNIX
EXPORT_C const long int *torch_tensor_get_sizes(const torch_tensor_t tensor);
#else
EXPORT_C const long long int *torch_tensor_get_sizes(const torch_tensor_t tensor);
#endif

/**
 * Function to delete a Torch Tensor to clean up
 * @param Torch Tensor to delete
 */
EXPORT_C void torch_tensor_delete(torch_tensor_t tensor);

/**
 * Overloads the assignment operator for Torch Tensor
 * @param input Tensor
 * @return copy of input Tensor
 */
EXPORT_C torch_tensor_t torch_tensor_assign(const torch_tensor_t input);

/**
 * Overloads the addition operator for two Torch Tensors
 * @param first Tensor to be added
 * @param second Tensor to be added
 * @return sum of the Tensors
 */
EXPORT_C torch_tensor_t torch_tensor_add(const torch_tensor_t tensor1,
                                         const torch_tensor_t tensor2);

/**
 * Overloads the subtraction operator for two Torch Tensors
 * @param first Tensor to be subtracted
 * @param second Tensor to be subtracted
 * @return difference of the Tensors
 */
EXPORT_C torch_tensor_t torch_tensor_subtract(const torch_tensor_t tensor1,
                                              const torch_tensor_t tensor2);

/**
 * Overloads the multiplication operator for two Torch Tensors
 * @param first Tensor to be multiplied
 * @param second Tensor to be multiplied
 * @return product of the Tensors
 */
EXPORT_C torch_tensor_t torch_tensor_multiply(const torch_tensor_t tensor1,
                                              const torch_tensor_t tensor2);

/**
 * Overloads the premultiplication operator for a scalar and a Torch Tensor
 * @param scalar to multiply by
 * @param Tensor to be multiplied
 * @return product of the scalar and Tensor
 */
EXPORT_C torch_tensor_t torch_tensor_premultiply(const float scalar,
                                                 const torch_tensor_t tensor);

/**
 * Overloads the postmultiplication operator for a Torch Tensor and a scalar
 * @param Tensor to be multiplied
 * @param scalar to multiply by
 * @return product of the Tensor and scalar
 */
EXPORT_C torch_tensor_t torch_tensor_postmultiply(const torch_tensor_t tensor,
                                                  const float scalar);

/**
 * Overloads the division operator for two Torch Tensors
 * @param first Tensor to be divided
 * @param second Tensor to be divided
 * @return quotient of the Tensors
 */
EXPORT_C torch_tensor_t torch_tensor_divide(const torch_tensor_t tensor1,
                                            const torch_tensor_t tensor2);

/**
 * Overloads the post-division operator for a Torch Tensor and a scalar
 * @param Tensor to be divided
 * @param scalar to divide by
 * @return quotient of the Tensor and scalar
 */
EXPORT_C torch_tensor_t torch_tensor_postdivide(const torch_tensor_t tensor,
                                                const float scalar);

/**
 * Overloads the exponentiation operator for two Torch Tensors
 * @param Tensor to take the power of
 * @param scalar exponent
 * @return power of the Tensor
 */
EXPORT_C torch_tensor_t torch_tensor_power(const torch_tensor_t tensor,
                                           const float exponent);

// =====================================================================================
// Module API
// =====================================================================================
/**
 * Function to load in a Torch model from a TorchScript file and store in a
 * Torch Module
 * @param filename where TorchScript description of model is stored
 * @param device type used (cpu, CUDA, etc.)
 * @param device index for the CUDA case
 * @param whether gradient is required
 * @param whether model is being trained
 * @return Torch Module loaded in from file
 */
EXPORT_C torch_jit_script_module_t torch_jit_load(const char *filename,
                                                  const torch_device_t device_type,
                                                  const int device_index,
                                                  const bool requires_grad,
                                                  const bool is_training);

/**
 * Function to run the `forward` method of a Torch Module
 * @param Torch Module containing the model
 * @param vector of Torch Tensors as inputs to the model
 * @param number of input Tensors in the input vector
 * @param vector of Torch Tensors as outputs from running the model
 * @param number of output Tensors in the output vector
 * @param whether gradient is required
 */
EXPORT_C void torch_jit_module_forward(const torch_jit_script_module_t module,
                                       const torch_tensor_t *inputs, const int nin,
                                       torch_tensor_t *outputs, const int nout,
                                       const bool requires_grad);

/**
 * Function to delete a Torch Module to clean up
 * @param Torch Module to delete
 */
EXPORT_C void torch_jit_module_delete(torch_jit_script_module_t module);

#endif /* C_TORCH_H*/
