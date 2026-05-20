!| Module for the FTorch `torch_model` type and associated procedures.
!
!  * License
!    FTorch is released under an MIT license.
!    See the [LICENSE](https://github.com/Cambridge-ICCS/FTorch/blob/main/LICENSE)
!    file for details.
module ftorch_loss
  use, intrinsic :: iso_c_binding, only : c_null_ptr, c_ptr
  use ftorch_tensor, only: torch_tensor

  implicit none

  public

! contains

  ! TODO: Implement MSELoss

  ! TODO: Implement CrossEntropyLoss

  ! TODO: Implement the other losses found at https://docs.pytorch.org/cppdocs/api/nn/loss.html

end module ftorch_loss
