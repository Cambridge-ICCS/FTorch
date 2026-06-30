!| Module for the FTorch `torch_model` type and associated procedures.
!
!  * License
!    FTorch is released under an MIT license.
!    See the [LICENSE](https://github.com/Cambridge-ICCS/FTorch/blob/main/LICENSE)
!    file for details.
module ftorch_loss
  use, intrinsic :: iso_c_binding, only : c_null_ptr, c_ptr
  use ftorch_types, only: torch_kNone, torch_kMean, torch_kSum
  use ftorch_tensor, only: torch_tensor

  implicit none

  public

contains

  ! ============================================================================
  ! --- Procedures for evaluating specific loss functions
  ! ============================================================================

  !| Evaluate MSELoss
  !
  !  Note that the reduction type relates to the operation to perform within a (mini)batch. With
  !  torch_kNone, no reduction is applied and the loss tensor will have the same dimensions as the
  !  input tensor. If torch_kMean (default) or torch_kSum is applied then the loss tensor will differ
  !  in the first dimension (for the batch), which will be collapsed.
  !
  !  We refer to the PyTorch docs for the specifics of how this works for MSELoss
  !  https://docs.pytorch.org/docs/main/nn.functional.html#torch.nn.functional.mse_loss
  subroutine torch_loss_mse(loss_tensor, input_tensor, target_tensor, reduction_type)
    use, intrinsic :: iso_c_binding, only : c_associated, c_int
    type(torch_tensor), intent(inout) :: loss_tensor  !! Tensor to hold the loss value
    type(torch_tensor), intent(in) :: input_tensor  !! Input tensor to evaluate loss at
    type(torch_tensor), intent(in) :: target_tensor  !! Target tensor to evaluate loss against
    integer, optional, intent(in) :: reduction_type  !! Reduction type to use over batches (default: torch_kMean)

    integer(c_int) :: reduction_type_value

    interface
      subroutine torch_loss_mse_c(loss_tensor_c, input_tensor_c, target_tensor_c, &
          reduction_type_c) bind(c, name = 'torch_loss_mse')
        use, intrinsic :: iso_c_binding, only : c_ptr, c_int
        implicit none
        type(c_ptr), value, intent(in) :: loss_tensor_c
        type(c_ptr), value, intent(in) :: input_tensor_c
        type(c_ptr), value, intent(in) :: target_tensor_c
        integer(c_int), value, intent(in) :: reduction_type_c
      end subroutine torch_loss_mse_c
    end interface

    ! Process optional arguments
    if (.not. present(reduction_type)) then
      reduction_type_value = torch_kMean
    else
      reduction_type_value = reduction_type
    end if

    if (.not. c_associated(loss_tensor%p)) then
      write(*,*) "Error :: loss tensor has not been constructed"
      stop 1
    end if
    call torch_loss_mse_c(loss_tensor%p, input_tensor%p, target_tensor%p, reduction_type_value)
  end subroutine torch_loss_mse

  !> Evaluate CrossEntropyLoss
  !
  !  Note that the reduction type relates to the operation to perform within a (mini)batch. With
  !  torch_kNone, no reduction is applied and the loss tensor will have the same dimensions as the
  !  input tensor. If torch_kMean (default) or torch_kSum is applied then the loss tensor will differ
  !  in the first dimension (for the batch), which will be collapsed.
  !
  !  Note also that, by definition, the result of CrossEntropyLoss will collapse across the
  !  class dimension. This is true even for reduction torch_kNone. This is important to
  !  consider if setting the size/shape of a Fortran array to hold these results.
  !  For more details see https://docs.pytorch.org/docs/main/generated/torch.nn.CrossEntropyLoss.html
  !
  !  We refer to the PyTorch docs for the specifics of how this works for CrossEntropyLoss
  !  https://docs.pytorch.org/docs/main/nn.functional.html#torch.nn.functional.cross_entropy
  subroutine torch_loss_cross_entropy(loss_tensor, input_tensor, target_tensor, reduction_type)
    use, intrinsic :: iso_c_binding, only : c_associated, c_int
    type(torch_tensor), intent(inout) :: loss_tensor  !! Tensor to hold the loss value
    type(torch_tensor), intent(in) :: input_tensor  !! Input tensor to evaluate loss at
    type(torch_tensor), intent(in) :: target_tensor  !! Target tensor to evaluate loss against
    integer, optional, intent(in) :: reduction_type  !! Optional reduction type (default: torch_kMean)

    integer(c_int) :: reduction_type_value

    interface
      subroutine torch_loss_cross_entropy_c(loss_tensor_c, input_tensor_c, target_tensor_c, &
          reduction_type_c) bind(c, name = 'torch_loss_cross_entropy')
        use, intrinsic :: iso_c_binding, only : c_ptr, c_int
        implicit none
        type(c_ptr), value, intent(in) :: loss_tensor_c
        type(c_ptr), value, intent(in) :: input_tensor_c
        type(c_ptr), value, intent(in) :: target_tensor_c
        integer(c_int), value, intent(in) :: reduction_type_c
      end subroutine torch_loss_cross_entropy_c
    end interface

    ! Process optional arguments
    if (.not. present(reduction_type)) then
      reduction_type_value = torch_kMean
    else
      reduction_type_value = reduction_type
    end if

    if (.not. c_associated(loss_tensor%p)) then
      write(*,*) "Error :: loss tensor has not been constructed"
      stop 1
    end if
    call torch_loss_cross_entropy_c(loss_tensor%p, input_tensor%p, target_tensor%p, &
                                    reduction_type_value)
  end subroutine torch_loss_cross_entropy

end module ftorch_loss
