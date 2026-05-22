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

contains

  ! ============================================================================
  ! --- Procedures for evaluating specific loss functions
  ! ============================================================================

  !> Evaluate MSELoss
  ! TODO: Allow reductions
  subroutine torch_loss_mse(input_tensor, target_tensor, loss_tensor)
    use, intrinsic :: iso_c_binding, only : c_ptr
    type(torch_tensor), intent(in) :: input_tensor  !! Input tensor to evaluate loss at
    type(torch_tensor), intent(in) :: target_tensor  !! Target tensor to evaluate loss against
    type(torch_tensor), intent(inout) :: loss_tensor  !! Tensor to hold the loss value

    interface
      subroutine torch_loss_mse_c(input_tensor_c, target_tensor_c, loss_tensor_c) &
          bind(c, name = 'torch_loss_mse')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: input_tensor_c
        type(c_ptr), value, intent(in) :: target_tensor_c
        type(c_ptr), value, intent(in) :: loss_tensor_c
      end subroutine torch_loss_mse_c
    end interface

    call torch_loss_mse_c(input_tensor%p, target_tensor%p, loss_tensor%p)
  end subroutine torch_loss_mse

  ! TODO: Implement CrossEntropyLoss

  ! TODO: Implement the other losses found at https://docs.pytorch.org/cppdocs/api/nn/loss.html

end module ftorch_loss
