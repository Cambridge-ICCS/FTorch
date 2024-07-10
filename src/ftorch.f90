!| Main module for FTorch containing types and procedures.
!  Generated from `ftorch.fypp` using the [fypp Fortran preprocessor](https://fypp.readthedocs.io/en/stable/index.html).
!
!  * License  
!    FTorch is released under an MIT license.
!    See the [LICENSE](https://github.com/Cambridge-ICCS/FTorch/blob/main/LICENSE)
!    file for details.

module ftorch

  use, intrinsic :: iso_c_binding, only: c_int, c_int8_t, c_int16_t, c_int32_t, c_int64_t, c_int64_t, &
                                         c_float, c_double, c_char, c_ptr, c_null_ptr
  use, intrinsic :: iso_fortran_env, only: int8, int16, int32, int64, real32, real64

  implicit none

  !> Type for holding a torch neural net (nn.Module).
  type torch_model
    type(c_ptr) :: p = c_null_ptr  !! pointer to the neural net in memory
  end type torch_model

  !> Type for holding a Torch tensor.
  type torch_tensor
    type(c_ptr) :: p = c_null_ptr  !! pointer to the tensor in memory
  end type torch_tensor

  !| Enumerator for Torch data types  
  !  From c_torch.h (torch_data_t)  
  !  Note that 0 `torch_kUInt8` and 5 `torch_kFloat16` are not sypported in Fortran
  enum, bind(c)
    enumerator :: torch_kUInt8 = 0 ! not supported in Fortran
    enumerator :: torch_kInt8 = 1
    enumerator :: torch_kInt16 = 2
    enumerator :: torch_kInt32 = 3
    enumerator :: torch_kInt64 = 4
    enumerator :: torch_kFloat16 = 5 ! not supported in Fortran
    enumerator :: torch_kFloat32 = 6
    enumerator :: torch_kFloat64 = 7
  end enum


  !| Enumerator for Torch devices  
  !  From c_torch.h (torch_device_t)
  enum, bind(c)
    enumerator :: torch_kCPU = 0
    enumerator :: torch_kCUDA = 1
  end enum

  !> Interface for directing `torch_tensor_from_array` to possible input types and ranks
  interface torch_tensor_from_array
    module procedure torch_tensor_from_array_int8_1d
    module procedure torch_tensor_from_array_int8_2d
    module procedure torch_tensor_from_array_int8_3d
    module procedure torch_tensor_from_array_int8_4d
    module procedure torch_tensor_from_array_int16_1d
    module procedure torch_tensor_from_array_int16_2d
    module procedure torch_tensor_from_array_int16_3d
    module procedure torch_tensor_from_array_int16_4d
    module procedure torch_tensor_from_array_int32_1d
    module procedure torch_tensor_from_array_int32_2d
    module procedure torch_tensor_from_array_int32_3d
    module procedure torch_tensor_from_array_int32_4d
    module procedure torch_tensor_from_array_int64_1d
    module procedure torch_tensor_from_array_int64_2d
    module procedure torch_tensor_from_array_int64_3d
    module procedure torch_tensor_from_array_int64_4d
    module procedure torch_tensor_from_array_real32_1d
    module procedure torch_tensor_from_array_real32_2d
    module procedure torch_tensor_from_array_real32_3d
    module procedure torch_tensor_from_array_real32_4d
    module procedure torch_tensor_from_array_real64_1d
    module procedure torch_tensor_from_array_real64_2d
    module procedure torch_tensor_from_array_real64_3d
    module procedure torch_tensor_from_array_real64_4d
  end interface

  !> Interface for deleting generic torch objects
  interface torch_delete
    module procedure torch_model_delete
    module procedure torch_tensor_delete
    module procedure torch_tensor_array_delete
  end interface

  interface
    function torch_from_blob_c(data, ndims, tensor_shape, strides, dtype, &
                               device_type, device_index, &
                               requires_grad) result(tensor_p) &
                               bind(c, name = 'torch_from_blob')
      use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_ptr

      ! Arguments
      type(c_ptr), value, intent(in)    :: data
      integer(c_int), value, intent(in) :: ndims
      integer(c_int64_t), intent(in)    :: tensor_shape(*)
      integer(c_int64_t), intent(in)    :: strides(*)
      integer(c_int), value, intent(in) :: dtype
      integer(c_int), value, intent(in) :: device_type
      integer(c_int), value, intent(in) :: device_index
      logical(c_bool), value, intent(in) :: requires_grad
      type(c_ptr)                       :: tensor_p
    end function torch_from_blob_c
  end interface

contains

  !> Returns a tensor filled with the scalar value 0.
  subroutine torch_tensor_zeros(tensor, ndims, tensor_shape, dtype, &
                                device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t
    type(torch_tensor), intent(out) :: tensor     !! Returned tensor
    integer(c_int), intent(in)      :: ndims      !! Number of dimensions of the tensor
    integer(c_int64_t), intent(in)  :: tensor_shape(*)   !! Shape of the tensor
    integer(c_int), intent(in)      :: dtype      !! Data type of the tensor
    integer(c_int), intent(in)      :: device_type  !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index     !! device index to use for `torch_kCUDA` case
    logical(c_bool), optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor
    integer(c_int)                  :: device_index_value  !! device index used
    logical(c_bool)                 :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    interface
      function torch_zeros_c(ndims, tensor_shape, dtype, device_type, device_index, requires_grad) result(tensor) &
          bind(c, name = 'torch_zeros')
        use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_ptr
        integer(c_int), value, intent(in) :: ndims
        integer(c_int64_t), intent(in)    :: tensor_shape(*)
        integer(c_int), value, intent(in) :: dtype
        integer(c_int), value, intent(in) :: device_type
        integer(c_int), value, intent(in) :: device_index
        logical(c_bool), value, intent(in) :: requires_grad
        type(c_ptr)                       :: tensor
      end function torch_zeros_c
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
      requires_grad_value = logical(.false., c_bool)
    else
      requires_grad_value = requires_grad
    end if

    tensor%p = torch_zeros_c(ndims, tensor_shape, dtype, device_type,          &
                             device_index_value, requires_grad_value)
  end subroutine torch_tensor_zeros

  !> Returns a tensor filled with the scalar value 1.
  subroutine torch_tensor_ones(tensor, ndims, tensor_shape, dtype, &
                               device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t
    type(torch_tensor), intent(out) :: tensor     !! Returned tensor
    integer(c_int), intent(in)      :: ndims      !! Number of dimensions of the tensor
    integer(c_int64_t), intent(in)  :: tensor_shape(*)   !! Shape of the tensor
    integer(c_int), intent(in)      :: dtype      !! Data type of the tensor
    integer(c_int), intent(in)      :: device_type  !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index     !! device index to use for `torch_kCUDA` case
    logical(c_bool), optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor
    integer(c_int)                  :: device_index_value  !! device index used
    logical(c_bool)                 :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    interface
      function torch_ones_c(ndims, tensor_shape, dtype, device_type, device_index, requires_grad) result(tensor) &
          bind(c, name = 'torch_ones')
        use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_ptr
        integer(c_int), value, intent(in) :: ndims
        integer(c_int64_t), intent(in)    :: tensor_shape(*)
        integer(c_int), value, intent(in) :: dtype
        integer(c_int), value, intent(in) :: device_type
        integer(c_int), value, intent(in) :: device_index
        logical(c_bool), value, intent(in) :: requires_grad
        type(c_ptr)                       :: tensor
      end function torch_ones_c
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
      requires_grad_value = logical(.false., c_bool)
    else
      requires_grad_value = requires_grad
    end if

    tensor%p = torch_ones_c(ndims, tensor_shape, dtype, device_type,           &
                            device_index_value, requires_grad_value)
  end subroutine torch_tensor_ones

  ! Torch Tensor API
  !| Exposes the given data as a tensor without taking ownership of the original data.
  !  This routine will take an (i, j, k) array and return an (k, j, i) tensor.
  subroutine torch_tensor_from_blob(tensor, data, ndims, tensor_shape, layout, dtype, &
                                    device_type, device_index, &
                                    requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_ptr
    type(torch_tensor), intent(out) :: tensor     !! Returned tensor
    type(c_ptr), intent(in)         :: data       !! Pointer to data
    integer(c_int), intent(in)      :: ndims      !! Number of dimensions of the tensor
    integer(c_int64_t), intent(in)  :: tensor_shape(*)   !! Shape of the tensor
    integer(c_int), intent(in)      :: layout(*)  !! Layout for strides for accessing data
    integer(c_int), intent(in)      :: dtype      !! Data type of the tensor
    integer(c_int), intent(in)      :: device_type  !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index     !! device index to use for `torch_kCUDA` case
    logical(c_bool), optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    integer(c_int)                  :: i          !! loop index
    integer(c_int64_t)              :: strides(ndims) !! Strides for accessing data
    integer(c_int)                  :: device_index_value  !! device index used
    logical(c_bool)                 :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    if (.not. present(requires_grad)) then
      requires_grad_value = logical(.false., c_bool)
    else
      requires_grad_value = requires_grad
    end if

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * tensor_shape(layout(i - 1))
    end do

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    tensor%p = torch_from_blob_c(data, ndims, tensor_shape, strides, dtype,    &
                                 device_type, device_index_value,              &
                                 requires_grad_value)
  end subroutine torch_tensor_from_blob

  !> Prints the contents of a tensor.
  subroutine torch_tensor_print(tensor)
    type(torch_tensor), intent(in) :: tensor  !! Input tensor

    interface
      subroutine torch_tensor_print_c(tensor) &
          bind(c, name = 'torch_tensor_print')
        use, intrinsic :: iso_c_binding, only : c_ptr
        type(c_ptr), value, intent(in) :: tensor
      end subroutine torch_tensor_print_c
    end interface

    call torch_tensor_print_c(tensor%p)
  end subroutine torch_tensor_print

  !> Determines the device index of a tensor.
  function torch_tensor_get_device_index(tensor) result(device_index)
    use, intrinsic :: iso_c_binding, only : c_int
    type(torch_tensor), intent(in) :: tensor  !! Input tensor
    integer(c_int) :: device_index  !! Device index of tensor

    interface
      function torch_tensor_get_device_index_c(tensor) result(device_index) &
          bind(c, name = 'torch_tensor_get_device_index')
        use, intrinsic :: iso_c_binding, only : c_int, c_ptr
        type(c_ptr), value, intent(in) :: tensor
        integer(c_int) :: device_index
      end function torch_tensor_get_device_index_c
    end interface

    device_index = torch_tensor_get_device_index_c(tensor%p)
  end function torch_tensor_get_device_index

  !> Deallocates an array of tensors.
  subroutine torch_tensor_array_delete(tensor_array)
    type(torch_tensor), dimension(:), intent(inout) :: tensor_array
    integer :: i

    ! use bounds rather than (1, N) because it's safer
    do i = lbound(tensor_array, dim=1), ubound(tensor_array, dim=1)
      call torch_tensor_delete(tensor_array(i))
    end do
  end subroutine torch_tensor_array_delete

  !> Deallocates a tensor.
  subroutine torch_tensor_delete(tensor)
    type(torch_tensor), intent(inout) :: tensor

    interface
      subroutine torch_tensor_delete_c(tensor) &
          bind(c, name = 'torch_tensor_delete')
        use, intrinsic :: iso_c_binding, only : c_ptr
        type(c_ptr), value, intent(in) :: tensor
      end subroutine torch_tensor_delete_c
    end interface

    call torch_tensor_delete_c(tensor%p)
  end subroutine torch_tensor_delete

  ! Torch Model API
  !> Loads a TorchScript nn.module (pre-trained PyTorch model saved with TorchScript)
  subroutine torch_model_load(model, filename, device_type, device_index, requires_grad, is_training)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_null_char
    type(torch_model), intent(out)       :: model   !! Returned deserialized model
    character(*), intent(in)             :: filename !! Filename of saved TorchScript model
    integer(c_int), optional, intent(in) :: device_type !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor
    logical, optional, intent(in) :: is_training  !! Whether gradients need to be computed for the created tensor
    integer(c_int) :: device_type_value
    integer(c_int) :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor
    logical :: is_training_value  !! Whether the model is being trained, rather than evaluated

    interface
      function torch_jit_load_c(filename, device_type, device_index, requires_grad, is_training) result(model) &
          bind(c, name = 'torch_jit_load')
        use, intrinsic :: iso_c_binding, only : c_bool, c_char, c_int, c_ptr
        character(c_char), intent(in) :: filename(*)
        integer(c_int), value, intent(in)    :: device_type
        integer(c_int), value, intent(in)    :: device_index
        logical(c_bool), value, intent(in) :: requires_grad
        logical(c_bool), value, intent(in) :: is_training
        type(c_ptr)                   :: model
      end function torch_jit_load_c
    end interface

    ! Process optional arguments
    if (present(device_type)) then
      device_type_value = device_type
    else
      device_type_value = torch_kCPU
    endif
    if (present(device_index)) then
      device_index_value = device_index
    else if (device_type_value == torch_kCPU) then
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
    model%p = torch_jit_load_c(trim(adjustl(filename))//c_null_char,           &
                                device_type_value, device_index_value,         &
                                logical(requires_grad_value, c_bool),          &
                                logical(is_training_value, c_bool))
  end subroutine torch_model_load

  !> Performs a forward pass of the model with the input tensors
  subroutine torch_model_forward(model, input_tensors, output_tensors, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_ptr, c_int, c_loc
    type(torch_model), intent(in) :: model        !! Model
    type(torch_tensor), intent(in), dimension(:) :: input_tensors  !! Array of Input tensors
    type(torch_tensor), intent(in), dimension(:) :: output_tensors !! Returned output tensors
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    integer :: i
    integer(c_int) ::  n_inputs
    integer(c_int) ::  n_outputs
    type(c_ptr), dimension(size(input_tensors)), target  :: input_ptrs
    type(c_ptr), dimension(size(output_tensors)), target  :: output_ptrs

    interface
      subroutine torch_jit_model_forward_c(model, input_tensors, n_inputs, &
          output_tensors, n_outputs, requires_grad) &
          bind(c, name = 'torch_jit_module_forward')
        use, intrinsic :: iso_c_binding, only : c_bool, c_ptr, c_int
        type(c_ptr), value, intent(in) :: model
        type(c_ptr), value, intent(in) :: input_tensors
        integer(c_int), value, intent(in) :: n_inputs
        type(c_ptr), value, intent(in) :: output_tensors
        integer(c_int), value, intent(in) :: n_outputs
        logical(c_bool), value, intent(in) :: requires_grad
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

  !> Deallocates a TorchScript model
  subroutine torch_model_delete(model)
    type(torch_model), intent(in) :: model     !! Torch Model to deallocate

    interface
      subroutine torch_jit_model_delete_c(model) &
          bind(c, name = 'torch_jit_module_delete')
        use, intrinsic :: iso_c_binding, only : c_ptr
        type(c_ptr), value, intent(in) :: model
      end subroutine torch_jit_model_delete_c
    end interface

    call torch_jit_model_delete_c(model%p)
  end subroutine torch_model_delete

  !> Return a Torch tensor pointing to data_in array of rank 1 containing data of type `int8`
  subroutine torch_tensor_from_array_int8_1d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int8

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    integer(kind=int8), intent(in), target :: data_in(:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(1) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(1)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kInt8 !! Data type
    integer(c_int64_t)        :: strides(1)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 1                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_int8_1d

  !> Return a Torch tensor pointing to data_in array of rank 2 containing data of type `int8`
  subroutine torch_tensor_from_array_int8_2d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int8

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    integer(kind=int8), intent(in), target :: data_in(:,:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(2) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(2)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kInt8 !! Data type
    integer(c_int64_t)        :: strides(2)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 2                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_int8_2d

  !> Return a Torch tensor pointing to data_in array of rank 3 containing data of type `int8`
  subroutine torch_tensor_from_array_int8_3d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int8

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    integer(kind=int8), intent(in), target :: data_in(:,:,:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(3) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(3)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kInt8 !! Data type
    integer(c_int64_t)        :: strides(3)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 3                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_int8_3d

  !> Return a Torch tensor pointing to data_in array of rank 4 containing data of type `int8`
  subroutine torch_tensor_from_array_int8_4d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int8

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    integer(kind=int8), intent(in), target :: data_in(:,:,:,:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(4) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(4)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kInt8 !! Data type
    integer(c_int64_t)        :: strides(4)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 4                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_int8_4d

  !> Return a Torch tensor pointing to data_in array of rank 1 containing data of type `int16`
  subroutine torch_tensor_from_array_int16_1d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int16

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    integer(kind=int16), intent(in), target :: data_in(:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(1) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(1)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kInt16 !! Data type
    integer(c_int64_t)        :: strides(1)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 1                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_int16_1d

  !> Return a Torch tensor pointing to data_in array of rank 2 containing data of type `int16`
  subroutine torch_tensor_from_array_int16_2d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int16

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    integer(kind=int16), intent(in), target :: data_in(:,:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(2) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(2)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kInt16 !! Data type
    integer(c_int64_t)        :: strides(2)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 2                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_int16_2d

  !> Return a Torch tensor pointing to data_in array of rank 3 containing data of type `int16`
  subroutine torch_tensor_from_array_int16_3d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int16

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    integer(kind=int16), intent(in), target :: data_in(:,:,:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(3) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(3)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kInt16 !! Data type
    integer(c_int64_t)        :: strides(3)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 3                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_int16_3d

  !> Return a Torch tensor pointing to data_in array of rank 4 containing data of type `int16`
  subroutine torch_tensor_from_array_int16_4d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int16

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    integer(kind=int16), intent(in), target :: data_in(:,:,:,:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(4) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(4)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kInt16 !! Data type
    integer(c_int64_t)        :: strides(4)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 4                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_int16_4d

  !> Return a Torch tensor pointing to data_in array of rank 1 containing data of type `int32`
  subroutine torch_tensor_from_array_int32_1d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    integer(kind=int32), intent(in), target :: data_in(:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(1) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(1)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kInt32 !! Data type
    integer(c_int64_t)        :: strides(1)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 1                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_int32_1d

  !> Return a Torch tensor pointing to data_in array of rank 2 containing data of type `int32`
  subroutine torch_tensor_from_array_int32_2d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    integer(kind=int32), intent(in), target :: data_in(:,:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(2) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(2)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kInt32 !! Data type
    integer(c_int64_t)        :: strides(2)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 2                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_int32_2d

  !> Return a Torch tensor pointing to data_in array of rank 3 containing data of type `int32`
  subroutine torch_tensor_from_array_int32_3d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    integer(kind=int32), intent(in), target :: data_in(:,:,:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(3) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(3)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kInt32 !! Data type
    integer(c_int64_t)        :: strides(3)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 3                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_int32_3d

  !> Return a Torch tensor pointing to data_in array of rank 4 containing data of type `int32`
  subroutine torch_tensor_from_array_int32_4d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    integer(kind=int32), intent(in), target :: data_in(:,:,:,:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(4) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(4)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kInt32 !! Data type
    integer(c_int64_t)        :: strides(4)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 4                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_int32_4d

  !> Return a Torch tensor pointing to data_in array of rank 1 containing data of type `int64`
  subroutine torch_tensor_from_array_int64_1d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    integer(kind=int64), intent(in), target :: data_in(:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(1) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(1)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kInt64 !! Data type
    integer(c_int64_t)        :: strides(1)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 1                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_int64_1d

  !> Return a Torch tensor pointing to data_in array of rank 2 containing data of type `int64`
  subroutine torch_tensor_from_array_int64_2d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    integer(kind=int64), intent(in), target :: data_in(:,:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(2) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(2)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kInt64 !! Data type
    integer(c_int64_t)        :: strides(2)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 2                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_int64_2d

  !> Return a Torch tensor pointing to data_in array of rank 3 containing data of type `int64`
  subroutine torch_tensor_from_array_int64_3d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    integer(kind=int64), intent(in), target :: data_in(:,:,:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(3) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(3)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kInt64 !! Data type
    integer(c_int64_t)        :: strides(3)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 3                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_int64_3d

  !> Return a Torch tensor pointing to data_in array of rank 4 containing data of type `int64`
  subroutine torch_tensor_from_array_int64_4d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    integer(kind=int64), intent(in), target :: data_in(:,:,:,:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(4) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(4)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kInt64 !! Data type
    integer(c_int64_t)        :: strides(4)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 4                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_int64_4d

  !> Return a Torch tensor pointing to data_in array of rank 1 containing data of type `real32`
  subroutine torch_tensor_from_array_real32_1d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : real32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    real(kind=real32), intent(in), target :: data_in(:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(1) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(1)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kFloat32 !! Data type
    integer(c_int64_t)        :: strides(1)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 1                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_real32_1d

  !> Return a Torch tensor pointing to data_in array of rank 2 containing data of type `real32`
  subroutine torch_tensor_from_array_real32_2d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : real32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    real(kind=real32), intent(in), target :: data_in(:,:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(2) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(2)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kFloat32 !! Data type
    integer(c_int64_t)        :: strides(2)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 2                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_real32_2d

  !> Return a Torch tensor pointing to data_in array of rank 3 containing data of type `real32`
  subroutine torch_tensor_from_array_real32_3d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : real32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    real(kind=real32), intent(in), target :: data_in(:,:,:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(3) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(3)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kFloat32 !! Data type
    integer(c_int64_t)        :: strides(3)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 3                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_real32_3d

  !> Return a Torch tensor pointing to data_in array of rank 4 containing data of type `real32`
  subroutine torch_tensor_from_array_real32_4d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : real32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    real(kind=real32), intent(in), target :: data_in(:,:,:,:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(4) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(4)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kFloat32 !! Data type
    integer(c_int64_t)        :: strides(4)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 4                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_real32_4d

  !> Return a Torch tensor pointing to data_in array of rank 1 containing data of type `real64`
  subroutine torch_tensor_from_array_real64_1d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : real64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    real(kind=real64), intent(in), target :: data_in(:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(1) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(1)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kFloat64 !! Data type
    integer(c_int64_t)        :: strides(1)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 1                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_real64_1d

  !> Return a Torch tensor pointing to data_in array of rank 2 containing data of type `real64`
  subroutine torch_tensor_from_array_real64_2d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : real64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    real(kind=real64), intent(in), target :: data_in(:,:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(2) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(2)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kFloat64 !! Data type
    integer(c_int64_t)        :: strides(2)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 2                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_real64_2d

  !> Return a Torch tensor pointing to data_in array of rank 3 containing data of type `real64`
  subroutine torch_tensor_from_array_real64_3d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : real64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    real(kind=real64), intent(in), target :: data_in(:,:,:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(3) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(3)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kFloat64 !! Data type
    integer(c_int64_t)        :: strides(3)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 3                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_real64_3d

  !> Return a Torch tensor pointing to data_in array of rank 4 containing data of type `real64`
  subroutine torch_tensor_from_array_real64_4d(tensor, data_in, layout, &
                                                        c_device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_float, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : real64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor !! Returned tensor

    ! inputs
    real(kind=real64), intent(in), target :: data_in(:,:,:,:)   !! Input data that tensor will point at
    integer, intent(in)        :: layout(4) !! Control order of indices
    integer(c_int), intent(in) :: c_device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index    !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: c_tensor_shape(4)           !! Shape of the tensor
    integer(c_int), parameter :: c_dtype = torch_kFloat64 !! Data type
    integer(c_int64_t)        :: strides(4)                  !! Strides for accessing data
    integer(c_int), parameter :: ndims = 4                   !! Number of dimension of input data
    integer                   :: i
    integer(c_int)            :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    ! Process optional arguments
    if (present(device_index)) then
      device_index_value = device_index
    else if (c_device_type == torch_kCPU) then
      device_index_value = -1
    else
      device_index_value = 0
    endif

    if (.not. present(requires_grad)) then
      requires_grad_value = .false.
    else
      requires_grad_value = requires_grad
    end if

    c_tensor_shape = shape(data_in)

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i - 1)) * c_tensor_shape(layout(i - 1))
    end do

    tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape,        &
                                 strides, c_dtype, c_device_type,              &
                                 device_index_value,                           &
                                 logical(requires_grad_value, c_bool))

  end subroutine torch_tensor_from_array_real64_4d


end module ftorch
