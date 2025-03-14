!| Optimisers module for FTorch.
!
!  * License
!    FTorch is released under an MIT license.
!    See the [LICENSE](https://github.com/Cambridge-ICCS/FTorch/blob/main/LICENSE)
!    file for details.

module ftorch_optim

  use, intrinsic :: iso_c_binding, only: c_associated, c_null_ptr, c_ptr
  use, intrinsic :: iso_fortran_env, only: int32

  use ftorch, only: torch_tensor, ftorch_int

  implicit none

  public

  ! ============================================================================
  ! --- Derived types
  ! ============================================================================

  !> Type for holding a torch optimizer.
  type torch_optim
    type(c_ptr) :: p = c_null_ptr  !! pointer to the optimizer in memory
  end type torch_optim

contains

  ! ============================================================================
  ! --- FTorch Optimizers API
  ! ============================================================================

  !> Performs a forward pass of the model with the input tensors
  subroutine torch_optim_SGD(optim, parameters, learning_rate)
    use, intrinsic :: iso_c_binding, only : c_ptr, c_int, c_double, c_loc
    use, intrinsic :: iso_fortran_env, only : real64
    type(torch_optim), intent(out) :: optim
    type(torch_tensor), intent(in), dimension(:) :: parameters  !! Array of parameter tensors
    real(kind=real64), optional, intent(in) :: learning_rate  !! learning rate for the optimization algorithm
    real(kind=real64) :: learning_rate_value  !! learning rate for the optimization algorithm

    integer(ftorch_int) :: i
    integer(c_int)      :: n_params
    type(c_ptr), dimension(size(parameters)), target  :: parameter_ptrs

    interface
      function torch_optim_SGD_c(parameters, n_params, learning_rate) result(optim)  &
          bind(c, name = 'torch_optim_SGD')
        use, intrinsic :: iso_c_binding, only : c_ptr, c_int, c_double
        implicit none
        type(c_ptr), value, intent(in) :: parameters
        integer(c_int), value, intent(in) :: n_params
        real(c_double), value, intent(in) :: learning_rate
        type(c_ptr) :: optim
      end function torch_optim_SGD_c
    end interface

    n_params = size(parameters)

    if (.not. present(learning_rate)) then
      learning_rate_value = learning_rate
    else
      learning_rate_value = 0.001_real64
    end if

    ! Assign array of pointers to the parameters
    do i = 1, n_params
      parameter_ptrs(i) = parameters(i)%p
    end do

    optim%p = torch_optim_SGD_c(c_loc(parameter_ptrs), n_params, learning_rate)
  end subroutine torch_optim_SGD

  !> Deallocates a TorchScript optimizer
  subroutine torch_optim_SGD_delete(optim)
    type(torch_optim), intent(in) :: optim  !! Optimizer to deallocate

    interface
      subroutine torch_optim_SGD_delete_c(optim) &
          bind(c, name = 'torch_optim_SGD_delete')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: optim
      end subroutine torch_optim_SGD_delete_c
    end interface

    call torch_optim_SGD_delete_c(optim%p)
  end subroutine torch_optim_SGD_delete

end module ftorch_optim
