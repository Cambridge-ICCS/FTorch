!| Main module for FTorch containing types and procedures.
!
!  * License
!    FTorch is released under an MIT license.
!    See the [LICENSE](https://github.com/Cambridge-ICCS/FTorch/blob/main/LICENSE)
!    file for details.
module ftorch
  use ftorch_devices, only: torch_kCPU, torch_kCUDA, torch_kHIP, torch_kXPU, torch_kMPS
  use ftorch_types, only: torch_kInt8, torch_kInt16, torch_kInt32, torch_kInt64, &
                          torch_kFloat32, torch_kFloat64, ftorch_int
  use ftorch_tensor, only: assignment(=), operator(+), operator(-), operator(*), operator(/), &
                           operator(**), torch_tensor, torch_tensor_empty, torch_tensor_zeros, &
                           torch_tensor_ones, torch_tensor_from_array, torch_tensor_from_blob, &
                           torch_tensor_to, torch_tensor_sum, torch_tensor_mean, &
                           torch_tensor_backward, torch_tensor_get_gradient, torch_tensor_delete, &
                           torch_tensor_array_delete, torch_tensor_get_rank, &
                           torch_tensor_get_shape, torch_tensor_get_stride, &
                           torch_tensor_get_dtype, torch_tensor_get_device_type, &
                           torch_tensor_get_device_index, torch_tensor_requires_grad, &
                           torch_tensor_zero, torch_tensor_zero_grad, torch_tensor_print
  use ftorch_model, only: torch_model, torch_model_load, torch_model_forward, torch_model_delete, &
                          torch_model_print_parameters, torch_model_is_training

  implicit none

  public

  !> Interface for deleting generic torch objects
  interface torch_delete
    module procedure torch_model_delete
    module procedure torch_tensor_delete
    module procedure torch_tensor_array_delete
  end interface

end module ftorch
