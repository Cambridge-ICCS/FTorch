#ifndef C_TORCH_H
#define C_TORCH_H

#ifdef __cplusplus
#define EXPORT_C extern "C"
#else
#define EXPORT_C
#endif

#include <stdint.h>

// =============================================================================
// --- Typedefs
// =============================================================================

// Opaque pointer type alias for torch::jit::script::Module class
typedef void *torch_jit_script_module_t;

// Opaque pointer type alias for at::Tensor
typedef void *torch_tensor_t;

// Opaque pointer type alias for integer scalars
typedef void *torch_int_t;

// Opaque pointer type alias for float scalars
typedef void *torch_float_t;

// Type that represents size, strides and indexing on tensors
// (like std::size_t for standard containers)
//
// Torch is using internally int64_t [i.e. signed 64bit integer]
// We can as well to avoid portability problems
// (i.e. 64bit integer beeing long long on Windows)
//
typedef int64_t torch_size_t;

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
// NOTE: Defined in main CMakeLists and passed via preprocessor
typedef enum {
  torch_kCPU = GPU_DEVICE_NONE,
  torch_kCUDA = GPU_DEVICE_CUDA,
  torch_kHIP = GPU_DEVICE_HIP,
  torch_kXPU = GPU_DEVICE_XPU,
  torch_kMPS = GPU_DEVICE_MPS,
} torch_device_t;

// =============================================================================
// --- Functions for constructing tensors
// =============================================================================

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

// =============================================================================
// --- Functions for interrogating tensors
// =============================================================================

/**
 * Function to print out a Torch Tensor
 * @param Torch Tensor to print
 */
EXPORT_C void torch_tensor_print(const torch_tensor_t tensor);

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
EXPORT_C const torch_size_t *torch_tensor_get_sizes(const torch_tensor_t tensor);

/**
 * Function to determine the strides of a Torch Tensor
 * @param tensor Torch tensor
 * @return pointer to the strides array of the tensor
 */
EXPORT_C const torch_size_t *torch_tensor_get_stride(const torch_tensor_t tensor);

/**
 * Function to determine the data type of a Torch Tensor
 * @param Torch Tensor to determine the data type of
 * @return data type of the Torch Tensor represented as an enum
 */
EXPORT_C torch_data_t torch_tensor_get_dtype(const torch_tensor_t tensor);

/**
 * Function to determine the device type of a Torch Tensor
 * @param Torch Tensor to determine the device type of
 * @return device type of the Torch Tensor represented as an enum
 */
EXPORT_C torch_device_t torch_tensor_get_device_type(const torch_tensor_t tensor);

/**
 * Function to determine the device index of a Torch Tensor
 * @param Torch Tensor to determine the device index of
 * @return device index of the Torch Tensor
 */
EXPORT_C int torch_tensor_get_device_index(const torch_tensor_t tensor);

/**
 * Function to determine whether a Torch Tensor requires the autograd module
 * @param Torch Tensor to interrogate
 * @return whether the Torch Tensor requires autograd
 */
EXPORT_C bool torch_tensor_requires_grad(const torch_tensor_t tensor);

// =============================================================================
// --- Functions for deallocating tensors
// =============================================================================

/**
 * Function to delete a Torch Tensor to clean up
 * @param Torch Tensor to delete
 */
EXPORT_C void torch_tensor_delete(torch_tensor_t tensor);

// =====================================================================================
// --- Functions for manipulating tensors
// =====================================================================================

/**
 * Function to reset the values of a Torch Tensor to zero
 * @param Torch Tensor to zero the values of
 */
EXPORT_C void torch_tensor_zero(torch_tensor_t tensor);

/**
 * Function to move a tensor to a target tensor's device and dtype
 * @param Tensor to be moved
 * @param Tensor with the target device and dtype
 * @param if True and this copy is happening between CPU and GPU, the copy may occur
 * asynchronously
 */
EXPORT_C void torch_tensor_to(const torch_tensor_t source_tensor,
                              torch_tensor_t target_tensor, bool non_blocking);

// =============================================================================
// --- Operator overloads acting on tensors
// =============================================================================

/**
 * Overloads the assignment operator for Torch Tensor
 * @param output Tensor
 * @param input Tensor
 */
EXPORT_C void torch_tensor_assign(torch_tensor_t output, const torch_tensor_t input);

/**
 * Overloads the addition operator for two Torch Tensors
 * @param sum of the Tensors
 * @param first Tensor to be added
 * @param second Tensor to be added
 */
EXPORT_C void torch_tensor_add(torch_tensor_t, const torch_tensor_t tensor1,
                               const torch_tensor_t tensor2);

/**
 * Overloads the minus operator for a single Torch Tensor
 * @param the negative Tensor
 * @param Tensor to take the negative of
 */
EXPORT_C void torch_tensor_negative(torch_tensor_t output, const torch_tensor_t tensor);

/**
 * Overloads the subtraction operator for two Torch Tensors
 * @param output Tensor
 * @param first Tensor to be subtracted
 * @param second Tensor to be subtracted
 */
EXPORT_C void torch_tensor_subtract(torch_tensor_t output, const torch_tensor_t tensor1,
                                    const torch_tensor_t tensor2);

/**
 * Overloads the multiplication operator for two Torch Tensors
 * @param output Tensor
 * @param first Tensor to be multiplied
 * @param second Tensor to be multiplied
 */
EXPORT_C void torch_tensor_multiply(torch_tensor_t output, const torch_tensor_t tensor1,
                                    const torch_tensor_t tensor2);

/**
 * Overloads the division operator for two Torch Tensors.
 * @param output Tensor
 * @param first Tensor to be divided
 * @param second Tensor to be divided
 */
EXPORT_C void torch_tensor_divide(torch_tensor_t output, const torch_tensor_t tensor1,
                                  const torch_tensor_t tensor2);

/**
 * Overloads the exponentiation operator for a Torch Tensor and an integer exponent
 * @param output Tensor
 * @param Tensor to take the power of
 * @param integer exponent
 */
EXPORT_C void torch_tensor_power_int(torch_tensor_t output, const torch_tensor_t tensor,
                                     const torch_int_t exponent);

/**
 * Overloads the exponentiation operator for a Torch Tensor and a floating point
 * exponent
 * @param output Tensor
 * @param Tensor to take the power of
 * @param floating point exponent
 */
EXPORT_C void torch_tensor_power_float(torch_tensor_t output,
                                       const torch_tensor_t tensor,
                                       const torch_float_t exponent);

// ============================================================================
// --- Other operators for computations involving tensors
// ============================================================================

/**
 * Overloads the summation operator for a Torch Tensor
 * @param output Tensor
 * @param Tensor to sum the values of
 */
EXPORT_C void torch_tensor_sum(torch_tensor_t output, const torch_tensor_t tensor);

/**
 * Overloads the mean operator for a Torch Tensor
 * @param output Tensor
 * @param Tensor to take the mean over the values of
 */
EXPORT_C void torch_tensor_mean(torch_tensor_t output, const torch_tensor_t tensor);

// =============================================================================
// --- Functions related to automatic differentiation functionality for tensors
// =============================================================================

/**
 * Function to reset the gradient values of a Torch Tensor to zero
 * @param Torch Tensor to zero the gradient values of
 */
EXPORT_C void torch_tensor_zero_grad(torch_tensor_t tensor);

/**
 * Function to perform back-propagation on a Torch Tensor.
 * Note that the Tensor must have the requires_grad attribute set to true.
 * @param Tensor to perform back-propagation on
 * @param Tensor with an external gradient to supply for the back-propagation
 * @param whether the computational graph should be retained
 */
EXPORT_C void torch_tensor_backward(const torch_tensor_t tensor,
                                    const torch_tensor_t external_gradient,
                                    const bool retain_graph);

/**
 * Function to return the grad attribute of a Torch Tensor.
 * @param Tensor to get the gradient of
 * @param Tensor for the gradient
 */
EXPORT_C void torch_tensor_get_gradient(const torch_tensor_t tensor,
                                        torch_tensor_t gradient);

// =============================================================================
// --- Torch model API
// =============================================================================

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
