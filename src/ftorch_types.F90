!| Module for defining FTorch's enumerators for device and data types.
!
!  * License
!    FTorch is released under an MIT license.
!    See the [LICENSE](https://github.com/Cambridge-ICCS/FTorch/blob/main/LICENSE)
!    file for details.
module ftorch_types
  implicit none

  public

  !| Enumerator for Torch devices
  !  From c_torch.h (torch_device_t)
  !  NOTE: Defined in main CMakeLists and passed via preprocessor
  enum, bind(c)
    enumerator :: torch_kCPU = GPU_DEVICE_NONE
    enumerator :: torch_kCUDA = GPU_DEVICE_CUDA
    enumerator :: torch_kHIP = GPU_DEVICE_HIP
    enumerator :: torch_kXPU = GPU_DEVICE_XPU
    enumerator :: torch_kMPS = GPU_DEVICE_MPS
  end enum

  !| Enumerator for Torch data types
  !  From c_torch.h (torch_data_t)
  !  Note that 0 `torch_kUInt8` and 5 `torch_kFloat16` are not sypported in Fortran
  enum, bind(c)
    enumerator :: torch_kUInt8 = 0  ! not supported in Fortran
    enumerator :: torch_kInt8 = 1
    enumerator :: torch_kInt16 = 2
    enumerator :: torch_kInt32 = 3
    enumerator :: torch_kInt64 = 4
    enumerator :: torch_kFloat16 = 5  ! not supported in Fortran
    enumerator :: torch_kFloat32 = 6
    enumerator :: torch_kFloat64 = 7
  end enum

  ! Set integer size for FTorch library
  integer, parameter :: ftorch_int = int32

end module ftorch_types
