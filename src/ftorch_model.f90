!| Module for the FTorch `torch_model` type and associated procedures.
!
!  * License
!    FTorch is released under an MIT license.
!    See the [LICENSE](https://github.com/Cambridge-ICCS/FTorch/blob/main/LICENSE)
!    file for details.
module ftorch_model
  use, intrinsic :: iso_c_binding, only : c_null_ptr, c_ptr
  use ftorch_devices, only: torch_kCPU, torch_kCUDA, torch_kHIP, torch_kXPU, torch_kMPS
  use ftorch_types, only: ftorch_int
  use ftorch_tensor, only: torch_tensor

  implicit none

  public

  !> Type for holding a torch neural net (nn.Module).
  type torch_model
    type(c_ptr) :: p = c_null_ptr  !! pointer to the neural net in memory
  contains
    procedure :: print_parameters => torch_model_print_parameters
    procedure :: is_training => torch_model_is_training
    final :: torch_model_delete
  end type torch_model

contains

  ! ============================================================================
  ! --- Procedures for constructing tensors
  ! ============================================================================

  !> Loads a TorchScript nn.module (pre-trained PyTorch model saved with TorchScript)
  subroutine torch_model_load(model, filename, device_type, device_index, &
                              requires_grad, is_training)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_null_char
    type(torch_model), intent(out) :: model    !! Returned deserialized model
    character(*), intent(in) :: filename       !! Filename of saved TorchScript model
    integer(c_int), intent(in) :: device_type  !! Device type the tensor will live on (`torch_kCPU` or a GPU device type)
    integer(c_int), optional, intent(in) :: device_index  !! Device index for GPU devices
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor
    logical, optional, intent(in) :: is_training    !! Whether the model is being trained, rather than evaluated
    integer(c_int) :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor
    logical :: is_training_value  !! Whether the model is being trained, rather than evaluated

    interface
      function torch_jit_load_c(filename_c, device_type_c, device_index_c, &
                                requires_grad_c, is_training_c) result(model_c) &
          bind(c, name = 'torch_jit_load')
        use, intrinsic :: iso_c_binding, only : c_bool, c_char, c_int, c_ptr
        implicit none
        character(c_char), intent(in) :: filename_c(*)
        integer(c_int), value, intent(in)    :: device_type_c
        integer(c_int), value, intent(in)    :: device_index_c
        logical(c_bool), value, intent(in) :: requires_grad_c
        logical(c_bool), value, intent(in) :: is_training_c
        type(c_ptr)                   :: model_c
      end function torch_jit_load_c
    end interface

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    if (.not. present(is_training)) then
      is_training_value = .false.
    else
      is_training_value = is_training
    end if

    ! Need to append c_null_char at end of filename
    model%p = torch_jit_load_c(trim(adjustl(filename))//c_null_char, device_type, &
                               device_index_value, logical(requires_grad_value, c_bool), &
                               logical(is_training_value, c_bool))
  end subroutine torch_model_load

  ! ============================================================================
  ! --- Procedures for performing inference
  ! ============================================================================

  !> Performs a forward pass of the model with the input tensors
  subroutine torch_model_forward(model, input_tensors, output_tensors, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_ptr, c_int, c_loc
    type(torch_model), intent(in) :: model  !! Model
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
      subroutine torch_jit_model_forward_c(model_c, input_tensors_c, n_inputs_c, &
                                           output_tensors_c, n_outputs_c, requires_grad_c) &
          bind(c, name = 'torch_jit_module_forward')
        use, intrinsic :: iso_c_binding, only : c_bool, c_ptr, c_int
        implicit none
        type(c_ptr), value, intent(in) :: model_c
        type(c_ptr), value, intent(in) :: input_tensors_c
        integer(c_int), value, intent(in) :: n_inputs_c
        type(c_ptr), value, intent(in) :: output_tensors_c
        integer(c_int), value, intent(in) :: n_outputs_c
        logical(c_bool), value, intent(in) :: requires_grad_c
      end subroutine torch_jit_model_forward_c
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

    call torch_jit_model_forward_c(model%p, c_loc(input_ptrs), n_inputs,       &
                                   c_loc(output_ptrs), n_outputs,              &
                                   logical(requires_grad_value, c_bool))
  end subroutine torch_model_forward

  ! ============================================================================
  ! --- Procedures for interrogating tensors
  ! ============================================================================

  !| Prints the parameters associated with a model
  !  NOTE: While viewing parameters in this way can be helpful for small toy models, it will produce
  !        large amounts of output for models with many, large, or high-dimensional parameters. In
  !        particular, tensors of 3 or more dimensions will be represented in terms of 2D arrays.
  subroutine torch_model_print_parameters(self)
    class(torch_model), intent(in) :: self  !! Model to print the parameters of

    interface
      subroutine torch_jit_model_print_parameters_c(model_c) &
          bind(c, name = 'torch_jit_module_print_parameters')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: model_c
      end subroutine torch_jit_model_print_parameters_c
    end interface

    call torch_jit_model_print_parameters_c(self%p)
  end subroutine torch_model_print_parameters

  !> Determines whether a model is set up for training
  function torch_model_is_training(self) result(is_training)
    class(torch_model), intent(in) :: self  !! Model to query
    logical :: is_training                  !! Whether the model is set up for training

    interface
      function torch_jit_model_is_training_c(model_c) result(is_training_c) &
          bind(c, name = 'torch_jit_module_is_training')
        use, intrinsic :: iso_c_binding, only : c_bool, c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: model_c
        logical(c_bool) :: is_training_c
      end function torch_jit_model_is_training_c
    end interface

    is_training = torch_jit_model_is_training_c(self%p)
  end function torch_model_is_training

  ! ============================================================================
  ! --- Procedures for deallocating models
  ! ============================================================================

  !> Deallocates a TorchScript model
  subroutine torch_model_delete(model)
    use, intrinsic :: iso_c_binding, only : c_associated, c_null_ptr
    type(torch_model), intent(inout) :: model  !! Torch Model to deallocate

    interface
      subroutine torch_jit_model_delete_c(model_c) &
          bind(c, name = 'torch_jit_module_delete')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: model_c
      end subroutine torch_jit_model_delete_c
    end interface

    ! Call the destructor, if it hasn't already been called
    if (c_associated(model%p)) then
      call torch_jit_model_delete_c(model%p)
      model%p = c_null_ptr
    end if
  end subroutine torch_model_delete

end module ftorch_model
