!| Module for defining FTorch's enumerators for data types.
!
!  * License
!    FTorch is released under an MIT license.
!    See the [LICENSE](https://github.com/Cambridge-ICCS/FTorch/blob/main/LICENSE)
!    file for details.
module ftorch_types
  use, intrinsic :: iso_fortran_env, only: int32

  implicit none

  public

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
