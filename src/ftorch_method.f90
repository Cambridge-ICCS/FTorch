!| Module for the FTorch `torch_method` type and associated procedures.
!
!  * License
!    FTorch is released under an MIT license.
!    See the [LICENSE](https://github.com/Cambridge-ICCS/FTorch/blob/main/LICENSE)
!    file for details.
module ftorch_method
  use, intrinsic :: iso_c_binding, only : c_null_ptr, c_ptr
  use ftorch_devices, only: torch_kCPU, torch_kCUDA, torch_kHIP, torch_kXPU, torch_kMPS
  use ftorch_model, only: torch_model
  use ftorch_types, only: ftorch_int
  use ftorch_tensor, only: torch_tensor

  implicit none

  public

  !> Type for holding a Torch Method from a TorchScript model.
  type torch_method
    type(c_ptr) :: p = c_null_ptr  !! pointer to the method in memory
  contains
    final :: torch_method_delete
  end type torch_method

contains

  ! ============================================================================
  ! --- Procedures for getting methods from a model
  ! ============================================================================

  !> Loads a Torch Method from a TorchScript nn.module (pre-trained PyTorch model saved with TorchScript)
  subroutine torch_get_method(method, model, methodname)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_null_char
    type(torch_method), intent(out) :: method    !! Returned deserialized method
    type(torch_model), intent(in) :: model       !! Model to associate with the method
    character(*), intent(in) :: methodname       !! Name of the method
    integer(c_int) :: device_index_value

    interface
      function torch_jit_get_method_c(model_c, methodname_c) result(method_c) &
          bind(c, name = 'torch_jit_get_method')
        use, intrinsic :: iso_c_binding, only : c_bool, c_char, c_int, c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: model_c
        character(c_char), intent(in)  :: methodname_c(*)
        type(c_ptr)                    :: method_c
      end function torch_jit_get_method_c
    end interface

    ! Need to append c_null_char at end of methodname
    method%p = torch_jit_get_method_c(model%p, trim(adjustl(methodname))//c_null_char)
  end subroutine torch_get_method

  ! ============================================================================
  ! --- Procedures for performing inference
  ! ============================================================================

  !> Performs a forward pass of the method with the input tensors
  subroutine torch_method_call(method, input_tensors, output_tensors, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_ptr, c_int, c_loc
    type(torch_method), intent(in) :: method  !! Method
    type(torch_tensor), intent(in), dimension(:) :: input_tensors   !! Array of Input tensors
    type(torch_tensor), intent(in), dimension(:) :: output_tensors  !! Returned output tensors
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    integer(ftorch_int) :: i
    integer(c_int)      :: n_inputs
    integer(c_int)      :: n_outputs
    type(c_ptr), dimension(size(input_tensors)), target  :: input_ptrs
    type(c_ptr), dimension(size(output_tensors)), target  :: output_ptrs

    interface
      subroutine torch_jit_method_call_c(method_c, input_tensors_c, n_inputs_c, &
                                         output_tensors_c, n_outputs_c, requires_grad_c) &
          bind(c, name = 'torch_jit_method_call')
        use, intrinsic :: iso_c_binding, only : c_bool, c_ptr, c_int
        implicit none
        type(c_ptr), value, intent(in) :: method_c
        type(c_ptr), value, intent(in) :: input_tensors_c
        integer(c_int), value, intent(in) :: n_inputs_c
        type(c_ptr), value, intent(in) :: output_tensors_c
        integer(c_int), value, intent(in) :: n_outputs_c
        logical(c_bool), value, intent(in) :: requires_grad_c
      end subroutine torch_jit_method_call_c
    end interface

    n_inputs = size(input_tensors)
    n_outputs = size(output_tensors)

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    ! Assign array of pointers to the input tensors
    do i = 1, n_inputs
      input_ptrs(i) = input_tensors(i)%p
    end do

    ! Assign array of pointers to the output tensors
    do i = 1, n_outputs
      output_ptrs(i) = output_tensors(i)%p
    end do

    call torch_jit_method_call_c(method%p, c_loc(input_ptrs), n_inputs,       &
                                 c_loc(output_ptrs), n_outputs,              &
                                 logical(requires_grad_value, c_bool))
  end subroutine torch_method_call

  ! ============================================================================
  ! --- Procedures for deallocating methods
  ! ============================================================================

  !> Deallocates a TorchScript method
  subroutine torch_method_delete(method)
    use, intrinsic :: iso_c_binding, only : c_associated, c_null_ptr
    type(torch_method), intent(inout) :: method  !! Torch Method to deallocate

    interface
      subroutine torch_jit_method_delete_c(method_c) &
          bind(c, name = 'torch_jit_method_delete')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: method_c
      end subroutine torch_jit_method_delete_c
    end interface

    ! Call the destructor, if it hasn't already been called
    if (c_associated(method%p)) then
      call torch_jit_method_delete_c(method%p)
      method%p = c_null_ptr
    end if
  end subroutine torch_method_delete

end module ftorch_method
