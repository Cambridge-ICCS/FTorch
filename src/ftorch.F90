!| Main module for FTorch containing types and procedures.
!  Generated from `ftorch.fypp` using the [fypp Fortran preprocessor](https://fypp.readthedocs.io/en/stable/index.html).
!
!  * License
!    FTorch is released under an MIT license.
!    See the [LICENSE](https://github.com/Cambridge-ICCS/FTorch/blob/main/LICENSE)
!    file for details.

module ftorch

  use, intrinsic :: iso_c_binding, only: c_associated, c_null_ptr, c_ptr
  use, intrinsic :: iso_fortran_env, only: int32

  implicit none

  ! Set integer size for FTorch library
  integer, parameter :: ftorch_int = int32

  ! ============================================================================
  ! --- Derived types and enums
  ! ============================================================================

  !> Type for holding a torch neural net (nn.Module).
  type torch_model
    type(c_ptr) :: p = c_null_ptr  !! pointer to the neural net in memory
  end type torch_model

  !> Type for holding a Torch tensor.
  type torch_tensor
    type(c_ptr) :: p = c_null_ptr  !! pointer to the tensor in memory
  contains
    procedure :: get_rank => torch_tensor_get_rank
    procedure :: get_shape => torch_tensor_get_shape
    procedure :: get_dtype => torch_tensor_get_dtype
    procedure :: get_device_type => torch_tensor_get_device_type
    procedure :: get_device_index => torch_tensor_get_device_index
    final :: torch_tensor_delete
  end type torch_tensor

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

  !| Enumerator for Torch devices
  !  From c_torch.h (torch_device_t)
  !  NOTE: Defined in main CMakeLists and passed via preprocessor
  enum, bind(c)
    enumerator :: torch_kCPU = GPU_DEVICE_NONE
    enumerator :: torch_kCUDA = GPU_DEVICE_CUDA
    enumerator :: torch_kXPU = GPU_DEVICE_XPU
    enumerator :: torch_kMPS = GPU_DEVICE_MPS
  end enum

  ! ============================================================================
  ! --- Interfaces for core FTorch procedures
  ! ============================================================================

  !> Interface for directing `torch_tensor_from_array` to possible input types and ranks
  interface torch_tensor_from_array
    module procedure torch_tensor_from_array_int8_1d
    module procedure torch_tensor_from_array_int8_2d
    module procedure torch_tensor_from_array_int8_3d
    module procedure torch_tensor_from_array_int8_4d
    module procedure torch_tensor_from_array_int8_5d
    module procedure torch_tensor_from_array_int16_1d
    module procedure torch_tensor_from_array_int16_2d
    module procedure torch_tensor_from_array_int16_3d
    module procedure torch_tensor_from_array_int16_4d
    module procedure torch_tensor_from_array_int16_5d
    module procedure torch_tensor_from_array_int32_1d
    module procedure torch_tensor_from_array_int32_2d
    module procedure torch_tensor_from_array_int32_3d
    module procedure torch_tensor_from_array_int32_4d
    module procedure torch_tensor_from_array_int32_5d
    module procedure torch_tensor_from_array_int64_1d
    module procedure torch_tensor_from_array_int64_2d
    module procedure torch_tensor_from_array_int64_3d
    module procedure torch_tensor_from_array_int64_4d
    module procedure torch_tensor_from_array_int64_5d
    module procedure torch_tensor_from_array_real32_1d
    module procedure torch_tensor_from_array_real32_2d
    module procedure torch_tensor_from_array_real32_3d
    module procedure torch_tensor_from_array_real32_4d
    module procedure torch_tensor_from_array_real32_5d
    module procedure torch_tensor_from_array_real64_1d
    module procedure torch_tensor_from_array_real64_2d
    module procedure torch_tensor_from_array_real64_3d
    module procedure torch_tensor_from_array_real64_4d
    module procedure torch_tensor_from_array_real64_5d
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

      implicit none

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

  ! ============================================================================
  ! --- Interfaces for overloaded operators acting on tensors
  ! ============================================================================

  interface assignment (=)
    module procedure torch_tensor_assign
  end interface

  interface operator (+)
    module procedure torch_tensor_add
  end interface

  interface operator (-)
    module procedure torch_tensor_negative
    module procedure torch_tensor_subtract
  end interface

  interface operator (*)
    module procedure torch_tensor_multiply
    module procedure torch_tensor_premultiply_int8
    module procedure torch_tensor_postmultiply_int8
    module procedure torch_tensor_premultiply_int16
    module procedure torch_tensor_postmultiply_int16
    module procedure torch_tensor_premultiply_int32
    module procedure torch_tensor_postmultiply_int32
    module procedure torch_tensor_premultiply_int64
    module procedure torch_tensor_postmultiply_int64
    module procedure torch_tensor_premultiply_real32
    module procedure torch_tensor_postmultiply_real32
    module procedure torch_tensor_premultiply_real64
    module procedure torch_tensor_postmultiply_real64
  end interface

  interface
    subroutine torch_tensor_multiply_c(output_c, tensor1_c, tensor2_c) &
        bind(c, name = 'torch_tensor_multiply')
      use, intrinsic :: iso_c_binding, only : c_ptr
      implicit none
      type(c_ptr), value, intent(in) :: output_c
      type(c_ptr), value, intent(in) :: tensor1_c
      type(c_ptr), value, intent(in) :: tensor2_c
    end subroutine torch_tensor_multiply_c
  end interface

  interface operator (/)
    module procedure torch_tensor_divide
    module procedure torch_tensor_postdivide_int8
    module procedure torch_tensor_postdivide_int16
    module procedure torch_tensor_postdivide_int32
    module procedure torch_tensor_postdivide_int64
    module procedure torch_tensor_postdivide_real32
    module procedure torch_tensor_postdivide_real64
  end interface

  interface
    subroutine torch_tensor_divide_c(output_c, tensor1_c, tensor2_c) &
        bind(c, name = 'torch_tensor_divide')
      use, intrinsic :: iso_c_binding, only : c_ptr
      implicit none
      type(c_ptr), value, intent(in) :: output_c
      type(c_ptr), value, intent(in) :: tensor1_c
      type(c_ptr), value, intent(in) :: tensor2_c
    end subroutine torch_tensor_divide_c
  end interface

  interface operator (**)
    module procedure torch_tensor_power_int8
    module procedure torch_tensor_power_int16
    module procedure torch_tensor_power_int32
    module procedure torch_tensor_power_int64
    module procedure torch_tensor_power_real32
    module procedure torch_tensor_power_real64
  end interface

contains

  ! ============================================================================
  ! --- Procedures for constructing tensors
  ! ============================================================================

  !> Returns a tensor with uninitialised values.
  subroutine torch_tensor_empty(tensor, ndims, tensor_shape, dtype, &
                                device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t
    type(torch_tensor), intent(out) :: tensor     !! Returned tensor
    integer(c_int), intent(in)      :: ndims      !! Number of dimensions of the tensor
    integer(c_int64_t), intent(in)  :: tensor_shape(:)   !! Shape of the tensor
    integer(c_int), intent(in)      :: dtype      !! Data type of the tensor
    integer(c_int), intent(in)      :: device_type  !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor
    integer(c_int)                  :: device_index_value  !! device index used
    logical(c_bool)                 :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    interface
      function torch_empty_c(ndims, tensor_shape, dtype, device_type, &
          device_index, requires_grad) result(tensor) &
          bind(c, name = 'torch_empty')
        use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_ptr

        implicit none

        integer(c_int), value, intent(in) :: ndims
        integer(c_int64_t), intent(in)    :: tensor_shape(*)
        integer(c_int), value, intent(in) :: dtype
        integer(c_int), value, intent(in) :: device_type
        integer(c_int), value, intent(in) :: device_index
        logical(c_bool), value, intent(in) :: requires_grad
        type(c_ptr)                       :: tensor
      end function torch_empty_c
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

    tensor%p = torch_empty_c(ndims, tensor_shape, dtype, device_type,          &
                             device_index_value, requires_grad_value)
  end subroutine torch_tensor_empty

  !> Returns a tensor filled with the scalar value 0.
  subroutine torch_tensor_zeros(tensor, ndims, tensor_shape, dtype, &
                                device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t
    type(torch_tensor), intent(out) :: tensor     !! Returned tensor
    integer(c_int), intent(in)      :: ndims      !! Number of dimensions of the tensor
    integer(c_int64_t), intent(in)  :: tensor_shape(:)   !! Shape of the tensor
    integer(c_int), intent(in)      :: dtype      !! Data type of the tensor
    integer(c_int), intent(in)      :: device_type  !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor
    integer(c_int)                  :: device_index_value   !! device index used
    logical(c_bool)                 :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    interface
      function torch_zeros_c(ndims, tensor_shape, dtype, &
                             device_type, device_index, requires_grad) result(tensor) &
          bind(c, name = 'torch_zeros')
        use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_ptr

        implicit none

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
    integer(c_int64_t), intent(in)  :: tensor_shape(:)   !! Shape of the tensor
    integer(c_int), intent(in)      :: dtype        !! Data type of the tensor
    integer(c_int), intent(in)      :: device_type  !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor
    integer(c_int)                  :: device_index_value    !! device index used
    logical(c_bool)                 :: requires_grad_value   !! Whether gradients need to be computed for the created tensor

    interface
      function torch_ones_c(ndims, tensor_shape, dtype, &
                            device_type, device_index, requires_grad) result(tensor) &
          bind(c, name = 'torch_ones')
        use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_ptr

        implicit none

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

  !| Exposes the given data as a tensor without taking ownership of the original data.
  !  This routine will take an (i, j, k) array and return an (k, j, i) tensor.
  subroutine torch_tensor_from_blob(tensor, data, ndims, tensor_shape, layout, dtype, &
                                    device_type, device_index, &
                                    requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_ptr
    type(torch_tensor), intent(out) :: tensor     !! Returned tensor
    type(c_ptr), intent(in)         :: data       !! Pointer to data
    integer(c_int), intent(in)      :: ndims      !! Number of dimensions of the tensor
    integer(c_int64_t), intent(in)  :: tensor_shape(:)  !! Shape of the tensor
    integer(c_int), intent(in)      :: layout(:)  !! Layout for strides for accessing data
    integer(c_int), intent(in)      :: dtype      !! Data type of the tensor
    integer(c_int), intent(in)      :: device_type  !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    integer(c_int)                  :: i                    !! loop index
    integer(c_int64_t)              :: strides(ndims)       !! Strides for accessing data
    integer(c_int)                  :: device_index_value   !! device index used
    logical(c_bool)                 :: requires_grad_value  !! Whether gradients need to be computed for the created tensor

    if (.not. present(requires_grad)) then
      requires_grad_value = logical(.false., c_bool)
    else
      requires_grad_value = requires_grad
    end if

    strides(:) = 0
    do i = 1, ndims
      if (i == 1) then
        strides(layout(i)) = 1
      else
        strides(layout(i)) = strides(layout(i - 1)) * tensor_shape(layout(i - 1))
      end if
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

  !> Return a Torch tensor pointing to data_in array of rank 1 containing data of type `int8`
  subroutine torch_tensor_from_array_int8_1d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int8

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int8), intent(in), target :: data_in(:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(1)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(1)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kInt8  !! Data type
    integer(c_int64_t)        :: strides(1)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 1                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_int8_1d

  !> Return a Torch tensor pointing to data_in array of rank 2 containing data of type `int8`
  subroutine torch_tensor_from_array_int8_2d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int8

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int8), intent(in), target :: data_in(:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(2)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(2)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kInt8  !! Data type
    integer(c_int64_t)        :: strides(2)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 2                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_int8_2d

  !> Return a Torch tensor pointing to data_in array of rank 3 containing data of type `int8`
  subroutine torch_tensor_from_array_int8_3d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int8

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int8), intent(in), target :: data_in(:,:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(3)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(3)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kInt8  !! Data type
    integer(c_int64_t)        :: strides(3)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 3                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_int8_3d

  !> Return a Torch tensor pointing to data_in array of rank 4 containing data of type `int8`
  subroutine torch_tensor_from_array_int8_4d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int8

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int8), intent(in), target :: data_in(:,:,:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(4)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(4)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kInt8  !! Data type
    integer(c_int64_t)        :: strides(4)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 4                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_int8_4d

  !> Return a Torch tensor pointing to data_in array of rank 5 containing data of type `int8`
  subroutine torch_tensor_from_array_int8_5d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int8

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int8), intent(in), target :: data_in(:,:,:,:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(5)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(5)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kInt8  !! Data type
    integer(c_int64_t)        :: strides(5)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 5                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_int8_5d

  !> Return a Torch tensor pointing to data_in array of rank 1 containing data of type `int16`
  subroutine torch_tensor_from_array_int16_1d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int16

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int16), intent(in), target :: data_in(:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(1)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(1)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kInt16  !! Data type
    integer(c_int64_t)        :: strides(1)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 1                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_int16_1d

  !> Return a Torch tensor pointing to data_in array of rank 2 containing data of type `int16`
  subroutine torch_tensor_from_array_int16_2d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int16

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int16), intent(in), target :: data_in(:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(2)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(2)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kInt16  !! Data type
    integer(c_int64_t)        :: strides(2)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 2                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_int16_2d

  !> Return a Torch tensor pointing to data_in array of rank 3 containing data of type `int16`
  subroutine torch_tensor_from_array_int16_3d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int16

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int16), intent(in), target :: data_in(:,:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(3)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(3)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kInt16  !! Data type
    integer(c_int64_t)        :: strides(3)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 3                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_int16_3d

  !> Return a Torch tensor pointing to data_in array of rank 4 containing data of type `int16`
  subroutine torch_tensor_from_array_int16_4d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int16

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int16), intent(in), target :: data_in(:,:,:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(4)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(4)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kInt16  !! Data type
    integer(c_int64_t)        :: strides(4)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 4                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_int16_4d

  !> Return a Torch tensor pointing to data_in array of rank 5 containing data of type `int16`
  subroutine torch_tensor_from_array_int16_5d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int16

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int16), intent(in), target :: data_in(:,:,:,:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(5)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(5)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kInt16  !! Data type
    integer(c_int64_t)        :: strides(5)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 5                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_int16_5d

  !> Return a Torch tensor pointing to data_in array of rank 1 containing data of type `int32`
  subroutine torch_tensor_from_array_int32_1d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int32), intent(in), target :: data_in(:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(1)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(1)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kInt32  !! Data type
    integer(c_int64_t)        :: strides(1)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 1                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_int32_1d

  !> Return a Torch tensor pointing to data_in array of rank 2 containing data of type `int32`
  subroutine torch_tensor_from_array_int32_2d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int32), intent(in), target :: data_in(:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(2)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(2)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kInt32  !! Data type
    integer(c_int64_t)        :: strides(2)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 2                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_int32_2d

  !> Return a Torch tensor pointing to data_in array of rank 3 containing data of type `int32`
  subroutine torch_tensor_from_array_int32_3d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int32), intent(in), target :: data_in(:,:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(3)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(3)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kInt32  !! Data type
    integer(c_int64_t)        :: strides(3)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 3                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_int32_3d

  !> Return a Torch tensor pointing to data_in array of rank 4 containing data of type `int32`
  subroutine torch_tensor_from_array_int32_4d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int32), intent(in), target :: data_in(:,:,:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(4)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(4)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kInt32  !! Data type
    integer(c_int64_t)        :: strides(4)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 4                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_int32_4d

  !> Return a Torch tensor pointing to data_in array of rank 5 containing data of type `int32`
  subroutine torch_tensor_from_array_int32_5d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int32), intent(in), target :: data_in(:,:,:,:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(5)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(5)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kInt32  !! Data type
    integer(c_int64_t)        :: strides(5)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 5                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_int32_5d

  !> Return a Torch tensor pointing to data_in array of rank 1 containing data of type `int64`
  subroutine torch_tensor_from_array_int64_1d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int64), intent(in), target :: data_in(:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(1)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(1)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kInt64  !! Data type
    integer(c_int64_t)        :: strides(1)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 1                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_int64_1d

  !> Return a Torch tensor pointing to data_in array of rank 2 containing data of type `int64`
  subroutine torch_tensor_from_array_int64_2d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int64), intent(in), target :: data_in(:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(2)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(2)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kInt64  !! Data type
    integer(c_int64_t)        :: strides(2)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 2                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_int64_2d

  !> Return a Torch tensor pointing to data_in array of rank 3 containing data of type `int64`
  subroutine torch_tensor_from_array_int64_3d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int64), intent(in), target :: data_in(:,:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(3)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(3)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kInt64  !! Data type
    integer(c_int64_t)        :: strides(3)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 3                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_int64_3d

  !> Return a Torch tensor pointing to data_in array of rank 4 containing data of type `int64`
  subroutine torch_tensor_from_array_int64_4d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int64), intent(in), target :: data_in(:,:,:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(4)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(4)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kInt64  !! Data type
    integer(c_int64_t)        :: strides(4)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 4                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_int64_4d

  !> Return a Torch tensor pointing to data_in array of rank 5 containing data of type `int64`
  subroutine torch_tensor_from_array_int64_5d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : int64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int64), intent(in), target :: data_in(:,:,:,:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(5)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(5)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kInt64  !! Data type
    integer(c_int64_t)        :: strides(5)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 5                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_int64_5d

  !> Return a Torch tensor pointing to data_in array of rank 1 containing data of type `real32`
  subroutine torch_tensor_from_array_real32_1d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : real32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    real(kind=real32), intent(in), target :: data_in(:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(1)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(1)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kFloat32  !! Data type
    integer(c_int64_t)        :: strides(1)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 1                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_real32_1d

  !> Return a Torch tensor pointing to data_in array of rank 2 containing data of type `real32`
  subroutine torch_tensor_from_array_real32_2d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : real32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    real(kind=real32), intent(in), target :: data_in(:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(2)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(2)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kFloat32  !! Data type
    integer(c_int64_t)        :: strides(2)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 2                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_real32_2d

  !> Return a Torch tensor pointing to data_in array of rank 3 containing data of type `real32`
  subroutine torch_tensor_from_array_real32_3d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : real32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    real(kind=real32), intent(in), target :: data_in(:,:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(3)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(3)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kFloat32  !! Data type
    integer(c_int64_t)        :: strides(3)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 3                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_real32_3d

  !> Return a Torch tensor pointing to data_in array of rank 4 containing data of type `real32`
  subroutine torch_tensor_from_array_real32_4d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : real32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    real(kind=real32), intent(in), target :: data_in(:,:,:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(4)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(4)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kFloat32  !! Data type
    integer(c_int64_t)        :: strides(4)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 4                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_real32_4d

  !> Return a Torch tensor pointing to data_in array of rank 5 containing data of type `real32`
  subroutine torch_tensor_from_array_real32_5d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : real32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    real(kind=real32), intent(in), target :: data_in(:,:,:,:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(5)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(5)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kFloat32  !! Data type
    integer(c_int64_t)        :: strides(5)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 5                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_real32_5d

  !> Return a Torch tensor pointing to data_in array of rank 1 containing data of type `real64`
  subroutine torch_tensor_from_array_real64_1d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : real64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    real(kind=real64), intent(in), target :: data_in(:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(1)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(1)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kFloat64  !! Data type
    integer(c_int64_t)        :: strides(1)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 1                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_real64_1d

  !> Return a Torch tensor pointing to data_in array of rank 2 containing data of type `real64`
  subroutine torch_tensor_from_array_real64_2d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : real64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    real(kind=real64), intent(in), target :: data_in(:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(2)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(2)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kFloat64  !! Data type
    integer(c_int64_t)        :: strides(2)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 2                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_real64_2d

  !> Return a Torch tensor pointing to data_in array of rank 3 containing data of type `real64`
  subroutine torch_tensor_from_array_real64_3d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : real64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    real(kind=real64), intent(in), target :: data_in(:,:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(3)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(3)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kFloat64  !! Data type
    integer(c_int64_t)        :: strides(3)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 3                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_real64_3d

  !> Return a Torch tensor pointing to data_in array of rank 4 containing data of type `real64`
  subroutine torch_tensor_from_array_real64_4d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : real64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    real(kind=real64), intent(in), target :: data_in(:,:,:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(4)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(4)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kFloat64  !! Data type
    integer(c_int64_t)        :: strides(4)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 4                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_real64_4d

  !> Return a Torch tensor pointing to data_in array of rank 5 containing data of type `real64`
  subroutine torch_tensor_from_array_real64_5d(tensor, data_in, layout, &
                                                        device_type, device_index, requires_grad)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_loc
    use, intrinsic :: iso_fortran_env, only : real64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    real(kind=real64), intent(in), target :: data_in(:,:,:,:,:)  !! Input data that tensor will point at
    integer(ftorch_int), intent(in) :: layout(5)  !! Control order of indices
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(c_int64_t)        :: tensor_shape(5)            !! Shape of the tensor
    integer(c_int), parameter :: dtype = torch_kFloat64  !! Data type
    integer(c_int64_t)        :: strides(5)                 !! Strides for accessing data
    integer(c_int), parameter :: ndims = 5                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_real64_5d


  ! ============================================================================
  ! --- Procedures for interrogating tensors
  ! ============================================================================

  !> Prints the contents of a tensor.
  subroutine torch_tensor_print(tensor)
    type(torch_tensor), intent(in) :: tensor  !! Input tensor

    interface
      subroutine torch_tensor_print_c(tensor) &
          bind(c, name = 'torch_tensor_print')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor
      end subroutine torch_tensor_print_c
    end interface

    call torch_tensor_print_c(tensor%p)
  end subroutine torch_tensor_print

  !> Determines the rank of a tensor.
  function torch_tensor_get_rank(self) result(rank)
    class(torch_tensor), intent(in) :: self
    integer(kind=int32) :: rank  !! rank of tensor

    interface
      function torch_tensor_get_rank_c(tensor) result(rank) &
          bind(c, name = 'torch_tensor_get_rank')
        use, intrinsic :: iso_c_binding, only : c_int, c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor
        integer(c_int) :: rank
      end function torch_tensor_get_rank_c
    end interface

    if (.not. c_associated(self%p)) then
      write(*,*) "Error :: tensor has not been constructed so its rank is unset"
      stop 1
    end if
    rank = torch_tensor_get_rank_c(self%p)
  end function torch_tensor_get_rank

  !> Determines the shape of a tensor.
  function torch_tensor_get_shape(self) result(sizes)
    use, intrinsic :: iso_c_binding, only : c_f_pointer, c_int, c_long, c_long_long, c_ptr
    class(torch_tensor), intent(in) :: self
#ifdef UNIX
    integer(kind=c_long), pointer :: sizes(:)  !! Pointer to tensor data
#else
    integer(kind=c_long_long), pointer :: sizes(:)  !! Pointer to tensor data
#endif
    integer(kind=int32) :: ndims(1)
    type(c_ptr) :: cptr

    interface
      function torch_tensor_get_sizes_c(tensor) result(sizes) &
          bind(c, name = 'torch_tensor_get_sizes')
        use, intrinsic :: iso_c_binding, only : c_int, c_long, c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor
        type(c_ptr) :: sizes
      end function torch_tensor_get_sizes_c
    end interface

    if (.not. c_associated(self%p)) then
      write(*,*) "Error :: tensor has not been constructed so its shape is unset"
      stop 1
    end if
    ndims(1) = self%get_rank()
    cptr = torch_tensor_get_sizes_c(self%p)
    call c_f_pointer(cptr, sizes, ndims)
  end function torch_tensor_get_shape

  !> Returns the data type of a tensor.
  function torch_tensor_get_dtype(self) result(dtype)
    use, intrinsic :: iso_c_binding, only : c_int
    class(torch_tensor), intent(in) :: self  !! Input tensor
    integer(c_int) :: dtype                  !! Data type of tensor

    interface
      function torch_tensor_get_dtype_c(tensor) result(dtype) &
          bind(c, name = 'torch_tensor_get_dtype')
        use, intrinsic :: iso_c_binding, only : c_int, c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor
        integer(c_int) :: dtype
      end function torch_tensor_get_dtype_c
    end interface

    if (.not. c_associated(self%p)) then
      write(*,*) "Error :: tensor has not been constructed so its data type is unset"
      stop 1
    end if
    dtype = torch_tensor_get_dtype_c(self%p)
  end function torch_tensor_get_dtype

  !> Returns the device type of a tensor.
  function torch_tensor_get_device_type(self) result(device_type)
    use, intrinsic :: iso_c_binding, only : c_int
    class(torch_tensor), intent(in) :: self  !! Input tensor
    integer(c_int) :: device_type            !! Device type of tensor

    interface
      function torch_tensor_get_device_type_c(tensor) result(device_type) &
          bind(c, name = 'torch_tensor_get_device_type')
        use, intrinsic :: iso_c_binding, only : c_int, c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor
        integer(c_int) :: device_type
      end function torch_tensor_get_device_type_c
    end interface

    if (.not. c_associated(self%p)) then
      write(*,*) "Error :: tensor has not been constructed so its device type is unset"
      stop 1
    end if
    device_type = torch_tensor_get_device_type_c(self%p)
  end function torch_tensor_get_device_type

  !> Determines the device index of a tensor.
  function torch_tensor_get_device_index(self) result(device_index)
    use, intrinsic :: iso_c_binding, only : c_int
    class(torch_tensor), intent(in) :: self  !! Input tensor
    integer(c_int) :: device_index           !! Device index of tensor

    interface
      function torch_tensor_get_device_index_c(tensor) result(device_index) &
          bind(c, name = 'torch_tensor_get_device_index')
        use, intrinsic :: iso_c_binding, only : c_int, c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor
        integer(c_int) :: device_index
      end function torch_tensor_get_device_index_c
    end interface

    if (.not. c_associated(self%p)) then
      write(*,*) "Error :: tensor has not been constructed so its device index is unset"
      stop 1
    end if
    device_index = torch_tensor_get_device_index_c(self%p)
  end function torch_tensor_get_device_index

  ! ============================================================================
  ! --- Procedures for deallocating tensors
  ! ============================================================================

  !> Deallocates an array of tensors.
  subroutine torch_tensor_array_delete(tensor_array)
    type(torch_tensor), dimension(:), intent(inout) :: tensor_array
    integer(ftorch_int) :: i

    ! use bounds rather than (1, N) because it's safer
    do i = lbound(tensor_array, dim=1), ubound(tensor_array, dim=1)
      call torch_tensor_delete(tensor_array(i))
    end do
  end subroutine torch_tensor_array_delete

  !> Deallocates a tensor.
  subroutine torch_tensor_delete(tensor)
    use, intrinsic :: iso_c_binding, only : c_associated, c_null_ptr
    type(torch_tensor), intent(inout) :: tensor

    interface
      subroutine torch_tensor_delete_c(tensor) &
          bind(c, name = 'torch_tensor_delete')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor
      end subroutine torch_tensor_delete_c
    end interface

    ! Call the destructor, if it hasn't already been called
    if (c_associated(tensor%p)) then
      call torch_tensor_delete_c(tensor%p)
      tensor%p = c_null_ptr
    end if
  end subroutine torch_tensor_delete

  ! ============================================================================
  ! --- Overloaded operators acting on tensors
  ! ============================================================================

  !> Overloads assignment operator for tensors.
  subroutine torch_tensor_assign(output, input)
    use, intrinsic :: iso_c_binding, only : c_associated
    type(torch_tensor), intent(inout) :: output
    type(torch_tensor), intent(in) :: input

    interface
      subroutine torch_tensor_assign_c(output_c, input_c) bind(c, name = 'torch_tensor_assign')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: output_c
        type(c_ptr), value, intent(in) :: input_c
      end subroutine torch_tensor_assign_c
    end interface

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, input%get_rank(), input%get_shape(), input%get_dtype(), &
                              input%get_device_type(), input%get_device_index())
    else if (input%get_device_type() /= output%get_device_type()) then
      write(*,*) "Error :: cannot assign tensors with different device types"
      stop 1
    end if
    call torch_tensor_assign_c(output%p, input%p)
  end subroutine torch_tensor_assign

  !> Overloads addition operator for two tensors.
  function torch_tensor_add(tensor1, tensor2) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated
    type(torch_tensor), intent(in) :: tensor1
    type(torch_tensor), intent(in) :: tensor2
    type(torch_tensor) :: output

    interface
      subroutine torch_tensor_add_c(output_c, tensor1_c, tensor2_c) &
          bind(c, name = 'torch_tensor_add')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor1_c
        type(c_ptr), value, intent(in) :: tensor2_c
        type(c_ptr), value, intent(in) :: output_c
      end subroutine torch_tensor_add_c
    end interface

    if (tensor1%get_device_type() /= tensor2%get_device_type()) then
      write(*,*) "Error :: cannot add tensors with different device types"
      stop 1
    end if
    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor1%get_rank(), tensor1%get_shape(), &
                              tensor1%get_dtype(), tensor1%get_device_type(), &
                              tensor1%get_device_index())
    end if
    call torch_tensor_add_c(output%p,tensor1%p, tensor2%p)
  end function torch_tensor_add

  !> Overloads negative operator for a single tensor.
  function torch_tensor_negative(tensor) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated
    type(torch_tensor), intent(in) :: tensor
    type(torch_tensor) :: output

    interface
      subroutine torch_tensor_negative_c(output_c, tensor_c) bind(c, name = 'torch_tensor_negative')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor_c
        type(c_ptr), value, intent(in) :: output_c
      end subroutine torch_tensor_negative_c
    end interface

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if
    call torch_tensor_negative_c(output%p, tensor%p)
  end function torch_tensor_negative

  !> Overloads subtraction operator for two tensors.
  function torch_tensor_subtract(tensor1, tensor2) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated
    type(torch_tensor), intent(in) :: tensor1
    type(torch_tensor), intent(in) :: tensor2
    type(torch_tensor) :: output

    interface
      subroutine torch_tensor_subtract_c(output_c, tensor1_c, tensor2_c) &
          bind(c, name = 'torch_tensor_subtract')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: output_c
        type(c_ptr), value, intent(in) :: tensor1_c
        type(c_ptr), value, intent(in) :: tensor2_c
      end subroutine torch_tensor_subtract_c
    end interface

    if (tensor1%get_device_type() /= tensor2%get_device_type()) then
      write(*,*) "Error :: cannot subtract tensors with different device types"
      stop 1
    end if

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor1%get_rank(), tensor1%get_shape(), &
                              tensor1%get_dtype(), tensor1%get_device_type(), &
                              tensor1%get_device_index())
    end if
    call torch_tensor_subtract_c(output%p, tensor1%p, tensor2%p)
  end function torch_tensor_subtract

  !> Overloads multiplication operator for two tensors.
  function torch_tensor_multiply(tensor1, tensor2) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated
    type(torch_tensor), intent(in) :: tensor1
    type(torch_tensor), intent(in) :: tensor2
    type(torch_tensor) :: output

    if (tensor1%get_device_type() /= tensor2%get_device_type()) then
      write(*,*) "Error :: cannot multiply tensors with different device types"
      stop 1
    end if

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor1%get_rank(), tensor1%get_shape(), &
                              tensor1%get_dtype(), tensor1%get_device_type(), &
                              tensor1%get_device_index())
    end if
    call torch_tensor_multiply_c(output%p, tensor1%p, tensor2%p)
  end function torch_tensor_multiply

  !> Overloads multiplication operator for a scalar of type int8 and a tensor.
  function torch_tensor_premultiply_int8(scalar, tensor) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : int8
    integer(int8), target, intent(in) :: scalar
    type(torch_tensor), intent(in) :: tensor
    type(torch_tensor) :: wrk, output

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if

    ! Create a tensor with a single entry, the scalar pre-multiplier
    call torch_tensor_from_array(wrk, [scalar], [1], tensor%get_device_type(), &
                                 tensor%get_device_index())
    call torch_tensor_multiply_c(output%p, wrk%p, tensor%p)
  end function torch_tensor_premultiply_int8

  !> Overloads multiplication operator for a scalar of type int16 and a tensor.
  function torch_tensor_premultiply_int16(scalar, tensor) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : int16
    integer(int16), target, intent(in) :: scalar
    type(torch_tensor), intent(in) :: tensor
    type(torch_tensor) :: wrk, output

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if

    ! Create a tensor with a single entry, the scalar pre-multiplier
    call torch_tensor_from_array(wrk, [scalar], [1], tensor%get_device_type(), &
                                 tensor%get_device_index())
    call torch_tensor_multiply_c(output%p, wrk%p, tensor%p)
  end function torch_tensor_premultiply_int16

  !> Overloads multiplication operator for a scalar of type int32 and a tensor.
  function torch_tensor_premultiply_int32(scalar, tensor) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : int32
    integer(int32), target, intent(in) :: scalar
    type(torch_tensor), intent(in) :: tensor
    type(torch_tensor) :: wrk, output

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if

    ! Create a tensor with a single entry, the scalar pre-multiplier
    call torch_tensor_from_array(wrk, [scalar], [1], tensor%get_device_type(), &
                                 tensor%get_device_index())
    call torch_tensor_multiply_c(output%p, wrk%p, tensor%p)
  end function torch_tensor_premultiply_int32

  !> Overloads multiplication operator for a scalar of type int64 and a tensor.
  function torch_tensor_premultiply_int64(scalar, tensor) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : int64
    integer(int64), target, intent(in) :: scalar
    type(torch_tensor), intent(in) :: tensor
    type(torch_tensor) :: wrk, output

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if

    ! Create a tensor with a single entry, the scalar pre-multiplier
    call torch_tensor_from_array(wrk, [scalar], [1], tensor%get_device_type(), &
                                 tensor%get_device_index())
    call torch_tensor_multiply_c(output%p, wrk%p, tensor%p)
  end function torch_tensor_premultiply_int64

  !> Overloads multiplication operator for a scalar of type real32 and a tensor.
  function torch_tensor_premultiply_real32(scalar, tensor) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : real32
    real(real32), target, intent(in) :: scalar
    type(torch_tensor), intent(in) :: tensor
    type(torch_tensor) :: wrk, output

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if

    ! Create a tensor with a single entry, the scalar pre-multiplier
    call torch_tensor_from_array(wrk, [scalar], [1], tensor%get_device_type(), &
                                 tensor%get_device_index())
    call torch_tensor_multiply_c(output%p, wrk%p, tensor%p)
  end function torch_tensor_premultiply_real32

  !> Overloads multiplication operator for a scalar of type real64 and a tensor.
  function torch_tensor_premultiply_real64(scalar, tensor) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : real64
    real(real64), target, intent(in) :: scalar
    type(torch_tensor), intent(in) :: tensor
    type(torch_tensor) :: wrk, output

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if

    ! Create a tensor with a single entry, the scalar pre-multiplier
    call torch_tensor_from_array(wrk, [scalar], [1], tensor%get_device_type(), &
                                 tensor%get_device_index())
    call torch_tensor_multiply_c(output%p, wrk%p, tensor%p)
  end function torch_tensor_premultiply_real64


  !> Overloads multiplication operator for a tensor and a scalar of type int8.
  function torch_tensor_postmultiply_int8(tensor, scalar) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : int8
    type(torch_tensor), intent(in) :: tensor
    integer(int8), intent(in) :: scalar
    type(torch_tensor) :: wrk, output

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if

    ! Create a tensor with a single entry, the scalar post-multiplier
    call torch_tensor_from_array(wrk, [scalar], [1], tensor%get_device_type(), &
                                 tensor%get_device_index())
    call torch_tensor_multiply_c(output%p, tensor%p, wrk%p)
  end function torch_tensor_postmultiply_int8

  !> Overloads multiplication operator for a tensor and a scalar of type int16.
  function torch_tensor_postmultiply_int16(tensor, scalar) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : int16
    type(torch_tensor), intent(in) :: tensor
    integer(int16), intent(in) :: scalar
    type(torch_tensor) :: wrk, output

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if

    ! Create a tensor with a single entry, the scalar post-multiplier
    call torch_tensor_from_array(wrk, [scalar], [1], tensor%get_device_type(), &
                                 tensor%get_device_index())
    call torch_tensor_multiply_c(output%p, tensor%p, wrk%p)
  end function torch_tensor_postmultiply_int16

  !> Overloads multiplication operator for a tensor and a scalar of type int32.
  function torch_tensor_postmultiply_int32(tensor, scalar) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : int32
    type(torch_tensor), intent(in) :: tensor
    integer(int32), intent(in) :: scalar
    type(torch_tensor) :: wrk, output

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if

    ! Create a tensor with a single entry, the scalar post-multiplier
    call torch_tensor_from_array(wrk, [scalar], [1], tensor%get_device_type(), &
                                 tensor%get_device_index())
    call torch_tensor_multiply_c(output%p, tensor%p, wrk%p)
  end function torch_tensor_postmultiply_int32

  !> Overloads multiplication operator for a tensor and a scalar of type int64.
  function torch_tensor_postmultiply_int64(tensor, scalar) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : int64
    type(torch_tensor), intent(in) :: tensor
    integer(int64), intent(in) :: scalar
    type(torch_tensor) :: wrk, output

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if

    ! Create a tensor with a single entry, the scalar post-multiplier
    call torch_tensor_from_array(wrk, [scalar], [1], tensor%get_device_type(), &
                                 tensor%get_device_index())
    call torch_tensor_multiply_c(output%p, tensor%p, wrk%p)
  end function torch_tensor_postmultiply_int64

  !> Overloads multiplication operator for a tensor and a scalar of type real32.
  function torch_tensor_postmultiply_real32(tensor, scalar) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : real32
    type(torch_tensor), intent(in) :: tensor
    real(real32), intent(in) :: scalar
    type(torch_tensor) :: wrk, output

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if

    ! Create a tensor with a single entry, the scalar post-multiplier
    call torch_tensor_from_array(wrk, [scalar], [1], tensor%get_device_type(), &
                                 tensor%get_device_index())
    call torch_tensor_multiply_c(output%p, tensor%p, wrk%p)
  end function torch_tensor_postmultiply_real32

  !> Overloads multiplication operator for a tensor and a scalar of type real64.
  function torch_tensor_postmultiply_real64(tensor, scalar) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : real64
    type(torch_tensor), intent(in) :: tensor
    real(real64), intent(in) :: scalar
    type(torch_tensor) :: wrk, output

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if

    ! Create a tensor with a single entry, the scalar post-multiplier
    call torch_tensor_from_array(wrk, [scalar], [1], tensor%get_device_type(), &
                                 tensor%get_device_index())
    call torch_tensor_multiply_c(output%p, tensor%p, wrk%p)
  end function torch_tensor_postmultiply_real64

  !> Overloads division operator for two tensors.
  function torch_tensor_divide(tensor1, tensor2) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated
    type(torch_tensor), intent(in) :: tensor1
    type(torch_tensor), intent(in) :: tensor2
    type(torch_tensor) :: output

    if (tensor1%get_device_type() /= tensor2%get_device_type()) then
      write(*,*) "Error :: cannot divide tensors with different device types"
      stop 1
    end if

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor1%get_rank(), tensor1%get_shape(), &
                              tensor1%get_dtype(), tensor1%get_device_type(), &
                              tensor1%get_device_index())
    end if
    call torch_tensor_divide_c(output%p, tensor1%p, tensor2%p)
  end function torch_tensor_divide

  !> Overloads division operator for a tensor and a scalar of type int8.
  function torch_tensor_postdivide_int8(tensor, divisor) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : int8
    type(torch_tensor), intent(in) :: tensor
    integer(int8), intent(in) :: divisor
    type(torch_tensor) :: wrk, output

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if

    ! Create a tensor with a single entry, the scalar post-divisor
    call torch_tensor_from_array(wrk, [divisor], [1], tensor%get_device_type(), &
                                 tensor%get_device_index())
    call torch_tensor_divide_c(output%p, tensor%p, wrk%p)
  end function torch_tensor_postdivide_int8

  !> Overloads division operator for a tensor and a scalar of type int16.
  function torch_tensor_postdivide_int16(tensor, divisor) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : int16
    type(torch_tensor), intent(in) :: tensor
    integer(int16), intent(in) :: divisor
    type(torch_tensor) :: wrk, output

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if

    ! Create a tensor with a single entry, the scalar post-divisor
    call torch_tensor_from_array(wrk, [divisor], [1], tensor%get_device_type(), &
                                 tensor%get_device_index())
    call torch_tensor_divide_c(output%p, tensor%p, wrk%p)
  end function torch_tensor_postdivide_int16

  !> Overloads division operator for a tensor and a scalar of type int32.
  function torch_tensor_postdivide_int32(tensor, divisor) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : int32
    type(torch_tensor), intent(in) :: tensor
    integer(int32), intent(in) :: divisor
    type(torch_tensor) :: wrk, output

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if

    ! Create a tensor with a single entry, the scalar post-divisor
    call torch_tensor_from_array(wrk, [divisor], [1], tensor%get_device_type(), &
                                 tensor%get_device_index())
    call torch_tensor_divide_c(output%p, tensor%p, wrk%p)
  end function torch_tensor_postdivide_int32

  !> Overloads division operator for a tensor and a scalar of type int64.
  function torch_tensor_postdivide_int64(tensor, divisor) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : int64
    type(torch_tensor), intent(in) :: tensor
    integer(int64), intent(in) :: divisor
    type(torch_tensor) :: wrk, output

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if

    ! Create a tensor with a single entry, the scalar post-divisor
    call torch_tensor_from_array(wrk, [divisor], [1], tensor%get_device_type(), &
                                 tensor%get_device_index())
    call torch_tensor_divide_c(output%p, tensor%p, wrk%p)
  end function torch_tensor_postdivide_int64

  !> Overloads division operator for a tensor and a scalar of type real32.
  function torch_tensor_postdivide_real32(tensor, divisor) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : real32
    type(torch_tensor), intent(in) :: tensor
    real(real32), intent(in) :: divisor
    type(torch_tensor) :: wrk, output

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if

    ! Create a tensor with a single entry, the scalar post-divisor
    call torch_tensor_from_array(wrk, [divisor], [1], tensor%get_device_type(), &
                                 tensor%get_device_index())
    call torch_tensor_divide_c(output%p, tensor%p, wrk%p)
  end function torch_tensor_postdivide_real32

  !> Overloads division operator for a tensor and a scalar of type real64.
  function torch_tensor_postdivide_real64(tensor, divisor) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : real64
    type(torch_tensor), intent(in) :: tensor
    real(real64), intent(in) :: divisor
    type(torch_tensor) :: wrk, output

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if

    ! Create a tensor with a single entry, the scalar post-divisor
    call torch_tensor_from_array(wrk, [divisor], [1], tensor%get_device_type(), &
                                 tensor%get_device_index())
    call torch_tensor_divide_c(output%p, tensor%p, wrk%p)
  end function torch_tensor_postdivide_real64


  !> Overloads exponentiation operator for a tensor and a scalar of type `int8`
  function torch_tensor_power_int8(tensor, power) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : int8
    type(torch_tensor), intent(in) :: tensor
    integer(int8), target, intent(in) :: power
    type(torch_tensor) :: output

    interface
      subroutine torch_tensor_power_int_c(output_c, tensor_c, power_c) &
          bind(c, name = 'torch_tensor_power_int')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: output_c
        type(c_ptr), value, intent(in) :: tensor_c
        type(c_ptr), value, intent(in) :: power_c
      end subroutine torch_tensor_power_int_c
    end interface

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if
    call torch_tensor_power_int_c(output%p, tensor%p, c_loc(power))
  end function torch_tensor_power_int8

  !> Overloads exponentiation operator for a tensor and a scalar of type `int16`
  function torch_tensor_power_int16(tensor, power) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : int16
    type(torch_tensor), intent(in) :: tensor
    integer(int16), target, intent(in) :: power
    type(torch_tensor) :: output

    interface
      subroutine torch_tensor_power_int_c(output_c, tensor_c, power_c) &
          bind(c, name = 'torch_tensor_power_int')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: output_c
        type(c_ptr), value, intent(in) :: tensor_c
        type(c_ptr), value, intent(in) :: power_c
      end subroutine torch_tensor_power_int_c
    end interface

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if
    call torch_tensor_power_int_c(output%p, tensor%p, c_loc(power))
  end function torch_tensor_power_int16

  !> Overloads exponentiation operator for a tensor and a scalar of type `int32`
  function torch_tensor_power_int32(tensor, power) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : int32
    type(torch_tensor), intent(in) :: tensor
    integer(int32), target, intent(in) :: power
    type(torch_tensor) :: output

    interface
      subroutine torch_tensor_power_int_c(output_c, tensor_c, power_c) &
          bind(c, name = 'torch_tensor_power_int')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: output_c
        type(c_ptr), value, intent(in) :: tensor_c
        type(c_ptr), value, intent(in) :: power_c
      end subroutine torch_tensor_power_int_c
    end interface

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if
    call torch_tensor_power_int_c(output%p, tensor%p, c_loc(power))
  end function torch_tensor_power_int32

  !> Overloads exponentiation operator for a tensor and a scalar of type `int64`
  function torch_tensor_power_int64(tensor, power) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : int64
    type(torch_tensor), intent(in) :: tensor
    integer(int64), target, intent(in) :: power
    type(torch_tensor) :: output

    interface
      subroutine torch_tensor_power_int_c(output_c, tensor_c, power_c) &
          bind(c, name = 'torch_tensor_power_int')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: output_c
        type(c_ptr), value, intent(in) :: tensor_c
        type(c_ptr), value, intent(in) :: power_c
      end subroutine torch_tensor_power_int_c
    end interface

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if
    call torch_tensor_power_int_c(output%p, tensor%p, c_loc(power))
  end function torch_tensor_power_int64


  !> Overloads exponentiation operator for a tensor and a scalar of type `real32`
  function torch_tensor_power_real32(tensor, power) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : real32
    type(torch_tensor), intent(in) :: tensor
    real(kind=real32), target, intent(in) :: power
    type(torch_tensor) :: output

    interface
      subroutine torch_tensor_power_float_c(output_c, tensor_c, power_c) &
          bind(c, name = 'torch_tensor_power_float')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: output_c
        type(c_ptr), value, intent(in) :: tensor_c
        type(c_ptr), value, intent(in) :: power_c
      end subroutine torch_tensor_power_float_c
    end interface

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if
    call torch_tensor_power_float_c(output%p, tensor%p, c_loc(power))
  end function torch_tensor_power_real32

  !> Overloads exponentiation operator for a tensor and a scalar of type `real64`
  function torch_tensor_power_real64(tensor, power) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : real64
    type(torch_tensor), intent(in) :: tensor
    real(kind=real64), target, intent(in) :: power
    type(torch_tensor) :: output

    interface
      subroutine torch_tensor_power_float_c(output_c, tensor_c, power_c) &
          bind(c, name = 'torch_tensor_power_float')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: output_c
        type(c_ptr), value, intent(in) :: tensor_c
        type(c_ptr), value, intent(in) :: power_c
      end subroutine torch_tensor_power_float_c
    end interface

    if (.not. c_associated(output%p)) then
      ! TODO: Pass requires_grad argument
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), tensor%get_device_index())
    end if
    call torch_tensor_power_float_c(output%p, tensor%p, c_loc(power))
  end function torch_tensor_power_real64


  ! ============================================================================
  ! --- Torch Model API
  ! ============================================================================

  !> Loads a TorchScript nn.module (pre-trained PyTorch model saved with TorchScript)
  subroutine torch_model_load(model, filename, device_type, device_index, &
                              requires_grad, is_training)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_null_char
    type(torch_model), intent(out) :: model    !! Returned deserialized model
    character(*), intent(in) :: filename       !! Filename of saved TorchScript model
    integer(c_int), intent(in) :: device_type  !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer(c_int), optional, intent(in) :: device_index  !! device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor
    logical, optional, intent(in) :: is_training    !! Whether gradients need to be computed for the created tensor
    integer(c_int) :: device_index_value
    logical :: requires_grad_value  !! Whether gradients need to be computed for the created tensor
    logical :: is_training_value  !! Whether the model is being trained, rather than evaluated

    interface
      function torch_jit_load_c(filename, device_type, device_index, &
                                requires_grad, is_training) result(model) &
          bind(c, name = 'torch_jit_load')
        use, intrinsic :: iso_c_binding, only : c_bool, c_char, c_int, c_ptr
        implicit none
        character(c_char), intent(in) :: filename(*)
        integer(c_int), value, intent(in)    :: device_type
        integer(c_int), value, intent(in)    :: device_index
        logical(c_bool), value, intent(in) :: requires_grad
        logical(c_bool), value, intent(in) :: is_training
        type(c_ptr)                   :: model
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
      subroutine torch_jit_model_forward_c(model, input_tensors, n_inputs, &
                                           output_tensors, n_outputs, requires_grad) &
          bind(c, name = 'torch_jit_module_forward')
        use, intrinsic :: iso_c_binding, only : c_bool, c_ptr, c_int
        implicit none
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
    type(torch_model), intent(in) :: model  !! Torch Model to deallocate

    interface
      subroutine torch_jit_model_delete_c(model) &
          bind(c, name = 'torch_jit_module_delete')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: model
      end subroutine torch_jit_model_delete_c
    end interface

    call torch_jit_model_delete_c(model%p)
  end subroutine torch_model_delete

end module ftorch
