!| Module for defining FTorch's enumerators for device types.
!
!  * License
!    FTorch is released under an MIT license.
!    See the [LICENSE](https://github.com/Cambridge-ICCS/FTorch/blob/main/LICENSE)
!    file for details.
module ftorch_devices
  use, intrinsic :: iso_fortran_env, only: int32

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

end module ftorch_devices
