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

  public

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
    procedure :: get_stride => torch_tensor_get_stride
    procedure :: get_dtype => torch_tensor_get_dtype
    procedure :: get_device_type => torch_tensor_get_device_type
    procedure :: get_device_index => torch_tensor_get_device_index
    procedure :: requires_grad => torch_tensor_requires_grad
    procedure :: zero => torch_tensor_zero
    procedure :: zero_grad => torch_tensor_zero_grad
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
    enumerator :: torch_kHIP = GPU_DEVICE_HIP
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
    module procedure torch_tensor_from_array_int8_1d_default_layout
    module procedure torch_tensor_from_array_int8_2d_default_layout
    module procedure torch_tensor_from_array_int8_3d_default_layout
    module procedure torch_tensor_from_array_int8_4d_default_layout
    module procedure torch_tensor_from_array_int8_5d_default_layout
    module procedure torch_tensor_from_array_int16_1d_default_layout
    module procedure torch_tensor_from_array_int16_2d_default_layout
    module procedure torch_tensor_from_array_int16_3d_default_layout
    module procedure torch_tensor_from_array_int16_4d_default_layout
    module procedure torch_tensor_from_array_int16_5d_default_layout
    module procedure torch_tensor_from_array_int32_1d_default_layout
    module procedure torch_tensor_from_array_int32_2d_default_layout
    module procedure torch_tensor_from_array_int32_3d_default_layout
    module procedure torch_tensor_from_array_int32_4d_default_layout
    module procedure torch_tensor_from_array_int32_5d_default_layout
    module procedure torch_tensor_from_array_int64_1d_default_layout
    module procedure torch_tensor_from_array_int64_2d_default_layout
    module procedure torch_tensor_from_array_int64_3d_default_layout
    module procedure torch_tensor_from_array_int64_4d_default_layout
    module procedure torch_tensor_from_array_int64_5d_default_layout
    module procedure torch_tensor_from_array_real32_1d_default_layout
    module procedure torch_tensor_from_array_real32_2d_default_layout
    module procedure torch_tensor_from_array_real32_3d_default_layout
    module procedure torch_tensor_from_array_real32_4d_default_layout
    module procedure torch_tensor_from_array_real32_5d_default_layout
    module procedure torch_tensor_from_array_real64_1d_default_layout
    module procedure torch_tensor_from_array_real64_2d_default_layout
    module procedure torch_tensor_from_array_real64_3d_default_layout
    module procedure torch_tensor_from_array_real64_4d_default_layout
    module procedure torch_tensor_from_array_real64_5d_default_layout
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
      function torch_empty_c(ndims_c, tensor_shape_c, dtype_c, device_type_c, &
          device_index_c, requires_grad_c) result(tensor_c) &
          bind(c, name = 'torch_empty')
        use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_ptr

        implicit none

        integer(c_int), value, intent(in) :: ndims_c
        integer(c_int64_t), intent(in)    :: tensor_shape_c(*)
        integer(c_int), value, intent(in) :: dtype_c
        integer(c_int), value, intent(in) :: device_type_c
        integer(c_int), value, intent(in) :: device_index_c
        logical(c_bool), value, intent(in) :: requires_grad_c
        type(c_ptr)                       :: tensor_c
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
      function torch_zeros_c(ndims_c, tensor_shape_c, dtype_c, &
                             device_type_c, device_index_c, requires_grad_c) result(tensor_c) &
          bind(c, name = 'torch_zeros')
        use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_ptr

        implicit none

        integer(c_int), value, intent(in) :: ndims_c
        integer(c_int64_t), intent(in)    :: tensor_shape_c(*)
        integer(c_int), value, intent(in) :: dtype_c
        integer(c_int), value, intent(in) :: device_type_c
        integer(c_int), value, intent(in) :: device_index_c
        logical(c_bool), value, intent(in) :: requires_grad_c
        type(c_ptr)                       :: tensor_c
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
      function torch_ones_c(ndims_c, tensor_shape_c, dtype_c, &
                            device_type_c, device_index_c, requires_grad_c) result(tensor_c) &
          bind(c, name = 'torch_ones')
        use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t, c_ptr

        implicit none

        integer(c_int), value, intent(in) :: ndims_c
        integer(c_int64_t), intent(in)    :: tensor_shape_c(*)
        integer(c_int), value, intent(in) :: dtype_c
        integer(c_int), value, intent(in) :: device_type_c
        integer(c_int), value, intent(in) :: device_index_c
        logical(c_bool), value, intent(in) :: requires_grad_c
        type(c_ptr)                       :: tensor_c
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
    integer(c_int), parameter :: ndims = 5                  !! Number of dimension of input data

    tensor_shape = shape(data_in)

    call torch_tensor_from_blob(tensor, c_loc(data_in), ndims, tensor_shape, &
                                layout, dtype, device_type, device_index, &
                                requires_grad)

  end subroutine torch_tensor_from_array_real64_5d

  ! TODO: Avoid the following variant of torch_tensor_from_array by making the `layout` argument
  !       optional. The reason this has not been done already is that it would require either making
  !       the `device_type` argument optional (which we do not want to do) or break the API.

  !> Return a Torch tensor pointing to data_in array of rank 1 containing data of type `int8` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_int8_1d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : int8

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int8), intent(in), target :: data_in(:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(1)  !! Order of indices
    integer(c_int), parameter :: ndims = 1  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_int8_1d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 2 containing data of type `int8` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_int8_2d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : int8

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int8), intent(in), target :: data_in(:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(2)  !! Order of indices
    integer(c_int), parameter :: ndims = 2  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_int8_2d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 3 containing data of type `int8` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_int8_3d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : int8

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int8), intent(in), target :: data_in(:,:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(3)  !! Order of indices
    integer(c_int), parameter :: ndims = 3  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_int8_3d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 4 containing data of type `int8` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_int8_4d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : int8

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int8), intent(in), target :: data_in(:,:,:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(4)  !! Order of indices
    integer(c_int), parameter :: ndims = 4  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_int8_4d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 5 containing data of type `int8` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_int8_5d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : int8

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int8), intent(in), target :: data_in(:,:,:,:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(5)  !! Order of indices
    integer(c_int), parameter :: ndims = 5  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_int8_5d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 1 containing data of type `int16` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_int16_1d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : int16

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int16), intent(in), target :: data_in(:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(1)  !! Order of indices
    integer(c_int), parameter :: ndims = 1  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_int16_1d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 2 containing data of type `int16` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_int16_2d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : int16

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int16), intent(in), target :: data_in(:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(2)  !! Order of indices
    integer(c_int), parameter :: ndims = 2  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_int16_2d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 3 containing data of type `int16` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_int16_3d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : int16

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int16), intent(in), target :: data_in(:,:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(3)  !! Order of indices
    integer(c_int), parameter :: ndims = 3  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_int16_3d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 4 containing data of type `int16` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_int16_4d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : int16

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int16), intent(in), target :: data_in(:,:,:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(4)  !! Order of indices
    integer(c_int), parameter :: ndims = 4  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_int16_4d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 5 containing data of type `int16` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_int16_5d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : int16

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int16), intent(in), target :: data_in(:,:,:,:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(5)  !! Order of indices
    integer(c_int), parameter :: ndims = 5  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_int16_5d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 1 containing data of type `int32` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_int32_1d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : int32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int32), intent(in), target :: data_in(:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(1)  !! Order of indices
    integer(c_int), parameter :: ndims = 1  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_int32_1d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 2 containing data of type `int32` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_int32_2d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : int32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int32), intent(in), target :: data_in(:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(2)  !! Order of indices
    integer(c_int), parameter :: ndims = 2  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_int32_2d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 3 containing data of type `int32` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_int32_3d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : int32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int32), intent(in), target :: data_in(:,:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(3)  !! Order of indices
    integer(c_int), parameter :: ndims = 3  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_int32_3d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 4 containing data of type `int32` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_int32_4d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : int32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int32), intent(in), target :: data_in(:,:,:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(4)  !! Order of indices
    integer(c_int), parameter :: ndims = 4  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_int32_4d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 5 containing data of type `int32` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_int32_5d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : int32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int32), intent(in), target :: data_in(:,:,:,:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(5)  !! Order of indices
    integer(c_int), parameter :: ndims = 5  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_int32_5d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 1 containing data of type `int64` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_int64_1d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : int64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int64), intent(in), target :: data_in(:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(1)  !! Order of indices
    integer(c_int), parameter :: ndims = 1  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_int64_1d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 2 containing data of type `int64` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_int64_2d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : int64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int64), intent(in), target :: data_in(:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(2)  !! Order of indices
    integer(c_int), parameter :: ndims = 2  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_int64_2d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 3 containing data of type `int64` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_int64_3d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : int64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int64), intent(in), target :: data_in(:,:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(3)  !! Order of indices
    integer(c_int), parameter :: ndims = 3  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_int64_3d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 4 containing data of type `int64` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_int64_4d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : int64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int64), intent(in), target :: data_in(:,:,:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(4)  !! Order of indices
    integer(c_int), parameter :: ndims = 4  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_int64_4d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 5 containing data of type `int64` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_int64_5d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : int64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    integer(kind=int64), intent(in), target :: data_in(:,:,:,:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(5)  !! Order of indices
    integer(c_int), parameter :: ndims = 5  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_int64_5d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 1 containing data of type `real32` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_real32_1d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : real32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    real(kind=real32), intent(in), target :: data_in(:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(1)  !! Order of indices
    integer(c_int), parameter :: ndims = 1  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_real32_1d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 2 containing data of type `real32` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_real32_2d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : real32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    real(kind=real32), intent(in), target :: data_in(:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(2)  !! Order of indices
    integer(c_int), parameter :: ndims = 2  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_real32_2d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 3 containing data of type `real32` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_real32_3d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : real32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    real(kind=real32), intent(in), target :: data_in(:,:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(3)  !! Order of indices
    integer(c_int), parameter :: ndims = 3  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_real32_3d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 4 containing data of type `real32` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_real32_4d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : real32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    real(kind=real32), intent(in), target :: data_in(:,:,:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(4)  !! Order of indices
    integer(c_int), parameter :: ndims = 4  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_real32_4d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 5 containing data of type `real32` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_real32_5d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : real32

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    real(kind=real32), intent(in), target :: data_in(:,:,:,:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(5)  !! Order of indices
    integer(c_int), parameter :: ndims = 5  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_real32_5d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 1 containing data of type `real64` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_real64_1d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : real64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    real(kind=real64), intent(in), target :: data_in(:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(1)  !! Order of indices
    integer(c_int), parameter :: ndims = 1  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_real64_1d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 2 containing data of type `real64` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_real64_2d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : real64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    real(kind=real64), intent(in), target :: data_in(:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(2)  !! Order of indices
    integer(c_int), parameter :: ndims = 2  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_real64_2d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 3 containing data of type `real64` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_real64_3d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : real64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    real(kind=real64), intent(in), target :: data_in(:,:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(3)  !! Order of indices
    integer(c_int), parameter :: ndims = 3  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_real64_3d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 4 containing data of type `real64` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_real64_4d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : real64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    real(kind=real64), intent(in), target :: data_in(:,:,:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(4)  !! Order of indices
    integer(c_int), parameter :: ndims = 4  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_real64_4d_default_layout

  !> Return a Torch tensor pointing to data_in array of rank 5 containing data of type `real64` with default layout [1, 2, ..., n].
  subroutine torch_tensor_from_array_real64_5d_default_layout(tensor, data_in, &
                                                                       device_type, device_index, &
                                                                       requires_grad)
    use, intrinsic :: iso_c_binding, only : c_int
    use, intrinsic :: iso_fortran_env, only : real64

    ! output tensor
    type(torch_tensor), intent(out) :: tensor  !! Returned tensor

    ! inputs
    real(kind=real64), intent(in), target :: data_in(:,:,:,:,:)  !! Input data that tensor will point at
    integer(c_int), intent(in)    :: device_type    !! Device type the tensor will live on (`torch_kCPU` or `torch_kCUDA`)
    integer, optional, intent(in) :: device_index   !! Device index to use for `torch_kCUDA` case
    logical, optional, intent(in) :: requires_grad  !! Whether gradients need to be computed for the created tensor

    ! local data
    integer(ftorch_int)       :: layout(5)  !! Order of indices
    integer(c_int), parameter :: ndims = 5  !! Number of dimension of input data
    integer :: i

    ! Set the default tensor layout
    do i = 1, ndims
      layout(i) = i
    end do

    call torch_tensor_from_array(tensor, data_in, layout, device_type, device_index, requires_grad)

  end subroutine torch_tensor_from_array_real64_5d_default_layout


  ! ============================================================================
  ! --- Procedures for interrogating tensors
  ! ============================================================================

  !> Prints the contents of a tensor.
  subroutine torch_tensor_print(tensor)
    type(torch_tensor), intent(in) :: tensor  !! Tensor to print the contents of

    interface
      subroutine torch_tensor_print_c(tensor_c) &
          bind(c, name = 'torch_tensor_print')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor_c
      end subroutine torch_tensor_print_c
    end interface

    call torch_tensor_print_c(tensor%p)
  end subroutine torch_tensor_print

  !> Determines the rank of a tensor.
  function torch_tensor_get_rank(self) result(rank)
    class(torch_tensor), intent(in) :: self  !! Tensor to get the rank of
    integer(kind=int32) :: rank              !! Rank of tensor

    interface
      function torch_tensor_get_rank_c(tensor_c) result(rank_c) &
          bind(c, name = 'torch_tensor_get_rank')
        use, intrinsic :: iso_c_binding, only : c_int, c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor_c
        integer(c_int) :: rank_c
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
    use, intrinsic :: iso_c_binding, only : c_f_pointer, c_int, c_int64_t, c_ptr
    class(torch_tensor), intent(in) :: self         !! Tensor to get the shape of
    integer(kind=c_int64_t), pointer :: sizes(:)       !! Pointer to tensor data

    ! Local data
    integer(kind=int32) :: ndims(1)
    type(c_ptr) :: cptr

    interface
      function torch_tensor_get_sizes_c(tensor_c) result(sizes_c) &
          bind(c, name = 'torch_tensor_get_sizes')
        use, intrinsic :: iso_c_binding, only : c_int, c_long, c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor_c
        type(c_ptr) :: sizes_c
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

  !> Return the strides of the tensor
  function torch_tensor_get_stride(self) result(strides)
    use, intrinsic :: iso_c_binding, only : c_f_pointer, c_int, c_int64_t, c_ptr
    class(torch_tensor), intent(in) :: self         !! Tensor to get the strides of
    integer(kind=c_int64_t), pointer :: strides(:)      !! Pointer to tensor data

    ! Local data
    integer(kind=int32) :: ndims(1)
    type(c_ptr) :: cptr

    interface
      function torch_tensor_get_stride_c(tensor_c) result(strides_c) &
          bind(c, name = 'torch_tensor_get_stride')
        use, intrinsic :: iso_c_binding, only : c_int, c_long, c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor_c
        type(c_ptr) :: strides_c
      end function torch_tensor_get_stride_c
    end interface

    if (.not. c_associated(self%p)) then
      write(*,*) "Error :: tensor has not been constructed so its strides are unset"
      stop 1
    end if

    ndims(1) = self%get_rank()
    cptr = torch_tensor_get_stride_c(self%p)
    call c_f_pointer(cptr, strides, ndims)

  end function torch_tensor_get_stride

  !> Returns the data type of a tensor.
  function torch_tensor_get_dtype(self) result(dtype)
    use, intrinsic :: iso_c_binding, only : c_int
    class(torch_tensor), intent(in) :: self  !! Tensor to get the data type of
    integer(c_int) :: dtype                  !! Data type of tensor

    interface
      function torch_tensor_get_dtype_c(tensor_c) result(dtype_c) &
          bind(c, name = 'torch_tensor_get_dtype')
        use, intrinsic :: iso_c_binding, only : c_int, c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor_c
        integer(c_int) :: dtype_c
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
    class(torch_tensor), intent(in) :: self  !! Tensor to get the device type of
    integer(c_int) :: device_type            !! Device type of tensor

    interface
      function torch_tensor_get_device_type_c(tensor_c) result(device_type_c) &
          bind(c, name = 'torch_tensor_get_device_type')
        use, intrinsic :: iso_c_binding, only : c_int, c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor_c
        integer(c_int) :: device_type_c
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
    class(torch_tensor), intent(in) :: self  !! Tensor to get the device index of
    integer(c_int) :: device_index           !! Device index of tensor

    interface
      function torch_tensor_get_device_index_c(tensor_c) result(device_index_c) &
          bind(c, name = 'torch_tensor_get_device_index')
        use, intrinsic :: iso_c_binding, only : c_int, c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor_c
        integer(c_int) :: device_index_c
      end function torch_tensor_get_device_index_c
    end interface

    if (.not. c_associated(self%p)) then
      write(*,*) "Error :: tensor has not been constructed so its device index is unset"
      stop 1
    end if
    device_index = torch_tensor_get_device_index_c(self%p)
  end function torch_tensor_get_device_index

  !> Determines whether a tensor requires the autograd module.
  function torch_tensor_requires_grad(self) result(requires_grad)
    class(torch_tensor), intent(in) :: self  !! Tensor to query
    logical :: requires_grad                 !! Whether the tensor requires autograd

    interface
      function torch_tensor_requires_grad_c(tensor_c) result(requires_grad_c) &
          bind(c, name = 'torch_tensor_requires_grad')
        use, intrinsic :: iso_c_binding, only : c_bool, c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor_c
        logical(c_bool) :: requires_grad_c
      end function torch_tensor_requires_grad_c
    end interface

    requires_grad = torch_tensor_requires_grad_c(self%p)
  end function torch_tensor_requires_grad

  ! ============================================================================
  ! --- Procedures for deallocating tensors
  ! ============================================================================

  !> Deallocates an array of tensors.
  subroutine torch_tensor_array_delete(tensor_array)
    type(torch_tensor), dimension(:), intent(inout) :: tensor_array  !! Array of tensors to deallocate

    ! Local data
    integer(ftorch_int) :: i

    ! use bounds rather than (1, N) because it's safer
    do i = lbound(tensor_array, dim=1), ubound(tensor_array, dim=1)
      call torch_tensor_delete(tensor_array(i))
    end do
  end subroutine torch_tensor_array_delete

  !> Deallocates a tensor.
  subroutine torch_tensor_delete(tensor)
    use, intrinsic :: iso_c_binding, only : c_associated, c_null_ptr
    type(torch_tensor), intent(inout) :: tensor  !! Tensor to deallocate

    interface
      subroutine torch_tensor_delete_c(tensor_c) &
          bind(c, name = 'torch_tensor_delete')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor_c
      end subroutine torch_tensor_delete_c
    end interface

    ! Call the destructor, if it hasn't already been called
    if (c_associated(tensor%p)) then
      call torch_tensor_delete_c(tensor%p)
      tensor%p = c_null_ptr
    end if
  end subroutine torch_tensor_delete

  ! ============================================================================
  ! --- Procedures for manipulating tensors
  ! ============================================================================

  !> Fills a tensor with the scalar value 0.
  subroutine torch_tensor_zero(tensor)
    use, intrinsic :: iso_c_binding, only : c_associated
    class(torch_tensor), intent(inout) :: tensor !! Tensor whose values are to be zeroed

    interface
      subroutine torch_tensor_zero_c(tensor_c) bind(c, name = 'torch_tensor_zero')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor_c
      end subroutine torch_tensor_zero_c
    end interface

    if (.not. c_associated(tensor%p)) then
      write(*,*) "Error :: tensor must be constructed before zeroing values"
      stop 1
    end if
    call torch_tensor_zero_c(tensor%p)
  end subroutine torch_tensor_zero

  !> Moves a source_tensor tensor to a target tensor's device and dtype
  subroutine torch_tensor_to(source_tensor, target_tensor, non_blocking)
    use, intrinsic :: iso_c_binding, only : c_bool, c_int, c_int64_t
    type(torch_tensor), intent(in) :: source_tensor      !! Source tensor to be moved
    type(torch_tensor), intent(inout) :: target_tensor   !! Target tensor with the desired device and dtype
    logical, optional, intent(in) :: non_blocking        !! Whether to perform asynchronous copy
    logical(c_bool) :: non_blocking_value
    integer(c_int) :: source_rank, target_rank, i
    integer(c_int64_t), pointer :: source_shape(:), target_shape(:)

    interface
      subroutine torch_tensor_to_c(source_tensor_c, target_tensor_c, non_blocking_c) &
          bind(c, name = 'torch_tensor_to')
        use, intrinsic :: iso_c_binding, only : c_bool, c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: source_tensor_c
        type(c_ptr), value, intent(in) :: target_tensor_c
        logical(c_bool), value, intent(in) :: non_blocking_c
      end subroutine torch_tensor_to_c
    end interface

    ! Check for rank and shape consistency between the source and target tensors
    source_rank = source_tensor%get_rank()
    target_rank = target_tensor%get_rank()

    if (source_rank /= target_rank) then
      write(*,*) "Error in torch_tensor_to :: Cannot move source_tensor to target_tensor because the ranks do not match."
      write(*,*) "Source tensor rank:", source_rank, "Target tensor rank:", target_rank
      stop 1
    end if

    source_shape => source_tensor%get_shape()
    target_shape => target_tensor%get_shape()

    do i = 1, source_rank
      if (source_shape(i) /= target_shape(i)) then
        write(*,*) "Error in torch_tensor_to :: Cannot move source_tensor to target_tensor because the shapes do not match."
        write(*,*) "Dimension", i, "mismatch: source_tensor =", source_shape(i), &
            "Target =", target_shape(i)
        stop 1
      end if
    end do

    ! Process optional arguments
    if (present(non_blocking)) then
      non_blocking_value = non_blocking
    else
      non_blocking_value = .false.
    end if

    call torch_tensor_to_c(source_tensor%p, target_tensor%p, non_blocking_value)

  end subroutine torch_tensor_to

  ! ============================================================================
  ! --- Overloaded operators acting on tensors
  ! ============================================================================

  !> Overloads assignment operator for tensors.
  subroutine torch_tensor_assign(output, input)
    use, intrinsic :: iso_c_binding, only : c_associated
    type(torch_tensor), intent(in) :: input      !! Tensor whose values are to be used
    type(torch_tensor), intent(inout) :: output  !! Tensor to assign values to

    interface
      subroutine torch_tensor_assign_c(output_c, input_c) bind(c, name = 'torch_tensor_assign')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: output_c
        type(c_ptr), value, intent(in) :: input_c
      end subroutine torch_tensor_assign_c
    end interface

    if (.not. c_associated(output%p)) then
      call torch_tensor_empty(output, input%get_rank(), input%get_shape(), input%get_dtype(), &
                              input%get_device_type(), device_index=input%get_device_index(), &
                              requires_grad=input%requires_grad())
    else if (input%get_device_type() /= output%get_device_type()) then
      write(*,*) "Error :: cannot assign tensors with different device types"
      stop 1
    end if
    call torch_tensor_assign_c(output%p, input%p)
  end subroutine torch_tensor_assign

  !> Overloads addition operator for two tensors.
  function torch_tensor_add(tensor1, tensor2) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated
    type(torch_tensor), intent(in) :: tensor1  !! First tensor to be added
    type(torch_tensor), intent(in) :: tensor2  !! Second tensor to be added
    type(torch_tensor) :: output               !! Tensor to hold the sum

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
      call torch_tensor_empty(output, tensor1%get_rank(), tensor1%get_shape(), &
                              tensor1%get_dtype(), tensor1%get_device_type(), &
                              device_index=tensor1%get_device_index(), &
                              requires_grad=tensor1%requires_grad())
    end if
    call torch_tensor_add_c(output%p,tensor1%p, tensor2%p)
  end function torch_tensor_add

  !> Overloads negative operator for a single tensor.
  function torch_tensor_negative(tensor) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated
    type(torch_tensor), intent(in) :: tensor  !! Tensor to take the negative of
    type(torch_tensor) :: output              !! Tensor to hold the negative values

    interface
      subroutine torch_tensor_negative_c(output_c, tensor_c) bind(c, name = 'torch_tensor_negative')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor_c
        type(c_ptr), value, intent(in) :: output_c
      end subroutine torch_tensor_negative_c
    end interface

    if (.not. c_associated(output%p)) then
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), device_index=tensor%get_device_index(), &
                              requires_grad=tensor%requires_grad())
    end if
    call torch_tensor_negative_c(output%p, tensor%p)
  end function torch_tensor_negative

  !> Overloads subtraction operator for two tensors.
  function torch_tensor_subtract(tensor1, tensor2) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated
    type(torch_tensor), intent(in) :: tensor1  !! First tensor for the subtraction
    type(torch_tensor), intent(in) :: tensor2  !! Second tensor for the subtraction
    type(torch_tensor) :: output               !! Tensor to hold the difference

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
      call torch_tensor_empty(output, tensor1%get_rank(), tensor1%get_shape(), &
                              tensor1%get_dtype(), tensor1%get_device_type(), &
                              device_index=tensor1%get_device_index(), &
                              requires_grad=tensor1%requires_grad())
    end if
    call torch_tensor_subtract_c(output%p, tensor1%p, tensor2%p)
  end function torch_tensor_subtract

  !> Overloads multiplication operator for two tensors.
  function torch_tensor_multiply(tensor1, tensor2) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated
    type(torch_tensor), intent(in) :: tensor1  !! First tensor to be multiplied
    type(torch_tensor), intent(in) :: tensor2  !! Second tensor to be multiplied
    type(torch_tensor) :: output               !! Tensor to hold the product

    if (tensor1%get_device_type() /= tensor2%get_device_type()) then
      write(*,*) "Error :: cannot multiply tensors with different device types"
      stop 1
    end if

    if (.not. c_associated(output%p)) then
      call torch_tensor_empty(output, tensor1%get_rank(), tensor1%get_shape(), &
                              tensor1%get_dtype(), tensor1%get_device_type(), &
                              device_index=tensor1%get_device_index(), &
                              requires_grad=tensor1%requires_grad())
    end if
    call torch_tensor_multiply_c(output%p, tensor1%p, tensor2%p)
  end function torch_tensor_multiply

  !> Overloads division operator for two tensors.
  function torch_tensor_divide(tensor1, tensor2) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated
    type(torch_tensor), intent(in) :: tensor1  !! First tensor for the division
    type(torch_tensor), intent(in) :: tensor2  !! Second tensor for the division
    type(torch_tensor) :: output               !! Tensor to hold the quotient

    if (tensor1%get_device_type() /= tensor2%get_device_type()) then
      write(*,*) "Error :: cannot divide tensors with different device types"
      stop 1
    end if

    if (.not. c_associated(output%p)) then
      call torch_tensor_empty(output, tensor1%get_rank(), tensor1%get_shape(), &
                              tensor1%get_dtype(), tensor1%get_device_type(), &
                              device_index=tensor1%get_device_index(), &
                              requires_grad=tensor1%requires_grad())
    end if
    call torch_tensor_divide_c(output%p, tensor1%p, tensor2%p)
  end function torch_tensor_divide

  !> Overloads exponentiation operator for a tensor and a scalar of type `int8`
  function torch_tensor_power_int8(tensor, power) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : int8
    type(torch_tensor), intent(in) :: tensor                  !! Tensor to take the power of
    integer(int8), target, intent(in) :: power   !! Integer exponent
    type(torch_tensor) :: output                              !! Tensor to hold the exponentiation

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
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), device_index=tensor%get_device_index(), &
                              requires_grad=tensor%requires_grad())
    end if
    call torch_tensor_power_int_c(output%p, tensor%p, c_loc(power))
  end function torch_tensor_power_int8

  !> Overloads exponentiation operator for a tensor and a scalar of type `int16`
  function torch_tensor_power_int16(tensor, power) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : int16
    type(torch_tensor), intent(in) :: tensor                  !! Tensor to take the power of
    integer(int16), target, intent(in) :: power   !! Integer exponent
    type(torch_tensor) :: output                              !! Tensor to hold the exponentiation

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
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), device_index=tensor%get_device_index(), &
                              requires_grad=tensor%requires_grad())
    end if
    call torch_tensor_power_int_c(output%p, tensor%p, c_loc(power))
  end function torch_tensor_power_int16

  !> Overloads exponentiation operator for a tensor and a scalar of type `int32`
  function torch_tensor_power_int32(tensor, power) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : int32
    type(torch_tensor), intent(in) :: tensor                  !! Tensor to take the power of
    integer(int32), target, intent(in) :: power   !! Integer exponent
    type(torch_tensor) :: output                              !! Tensor to hold the exponentiation

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
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), device_index=tensor%get_device_index(), &
                              requires_grad=tensor%requires_grad())
    end if
    call torch_tensor_power_int_c(output%p, tensor%p, c_loc(power))
  end function torch_tensor_power_int32

  !> Overloads exponentiation operator for a tensor and a scalar of type `int64`
  function torch_tensor_power_int64(tensor, power) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : int64
    type(torch_tensor), intent(in) :: tensor                  !! Tensor to take the power of
    integer(int64), target, intent(in) :: power   !! Integer exponent
    type(torch_tensor) :: output                              !! Tensor to hold the exponentiation

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
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), device_index=tensor%get_device_index(), &
                              requires_grad=tensor%requires_grad())
    end if
    call torch_tensor_power_int_c(output%p, tensor%p, c_loc(power))
  end function torch_tensor_power_int64


  !> Overloads exponentiation operator for a tensor and a scalar of type `real32`
  function torch_tensor_power_real32(tensor, power) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : real32
    type(torch_tensor), intent(in) :: tensor                      !! Tensor to take the power of
    real(kind=real32), target, intent(in) :: power  !! Floating point exponent
    type(torch_tensor) :: output                                  !! Tensor to hold the exponentiation

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
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), device_index=tensor%get_device_index(), &
                              requires_grad=tensor%requires_grad())
    end if
    call torch_tensor_power_float_c(output%p, tensor%p, c_loc(power))
  end function torch_tensor_power_real32

  !> Overloads exponentiation operator for a tensor and a scalar of type `real64`
  function torch_tensor_power_real64(tensor, power) result(output)
    use, intrinsic :: iso_c_binding, only : c_associated, c_loc
    use, intrinsic :: iso_fortran_env, only : real64
    type(torch_tensor), intent(in) :: tensor                      !! Tensor to take the power of
    real(kind=real64), target, intent(in) :: power  !! Floating point exponent
    type(torch_tensor) :: output                                  !! Tensor to hold the exponentiation

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
      call torch_tensor_empty(output, tensor%get_rank(), tensor%get_shape(), tensor%get_dtype(), &
                              tensor%get_device_type(), device_index=tensor%get_device_index(), &
                              requires_grad=tensor%requires_grad())
    end if
    call torch_tensor_power_float_c(output%p, tensor%p, c_loc(power))
  end function torch_tensor_power_real64


  ! ============================================================================
  ! --- Other operators for computations involving tensors
  ! ============================================================================

  !> Overloads summation operator over the values in a tensor.
  subroutine torch_tensor_sum(output, tensor)
    use, intrinsic :: iso_c_binding, only : c_associated
    type(torch_tensor), intent(inout) :: output  !! Tensor holding the summed values
    type(torch_tensor), intent(in)    :: tensor  !! Tensor to sum the values of

    interface
      subroutine torch_tensor_sum_c(output_c, tensor_c) &
          bind(c, name = 'torch_tensor_sum')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: output_c
        type(c_ptr), value, intent(in) :: tensor_c
      end subroutine torch_tensor_sum_c
    end interface

    if (.not. c_associated(output%p)) then
      write(*,*) "Error :: output tensor has not been constructed"
      stop 1
    end if
    call torch_tensor_sum_c(output%p, tensor%p)
  end subroutine torch_tensor_sum

  !> Overloads mean operator over the values in a tensor.
  subroutine torch_tensor_mean(output, tensor)
    use, intrinsic :: iso_c_binding, only : c_associated
    type(torch_tensor), intent(inout) :: output  !! Tensor holding the averaged values
    type(torch_tensor), intent(in)    :: tensor  !! Tensor to average the values of

    interface
      subroutine torch_tensor_mean_c(output_c, tensor_c) &
          bind(c, name = 'torch_tensor_mean')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: output_c
        type(c_ptr), value, intent(in) :: tensor_c
      end subroutine torch_tensor_mean_c
    end interface

    if (.not. c_associated(output%p)) then
      write(*,*) "Error :: output tensor has not been constructed"
      stop 1
    end if
    call torch_tensor_mean_c(output%p, tensor%p)
  end subroutine torch_tensor_mean

  ! ============================================================================
  ! --- Procedures related to automatic differentation functionality for tensors
  ! ============================================================================

  !> Resets a tensor's gradient to zero.
  subroutine torch_tensor_zero_grad(tensor)
    use, intrinsic :: iso_c_binding, only : c_associated
    class(torch_tensor), intent(inout) :: tensor  !! Tensor to zero the gradient of

    interface
      subroutine torch_tensor_zero_grad_c(tensor_c) bind(c, name = 'torch_tensor_zero_grad')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor_c
      end subroutine torch_tensor_zero_grad_c
    end interface

    ! TODO: Call torch_tensor_get_gradient to check it exists?
    call torch_tensor_zero_grad_c(tensor%p)
  end subroutine torch_tensor_zero_grad

  !> Performs back-propagation on a Torch Tensor, given some external gradient.
  subroutine torch_tensor_backward(tensor, retain_graph)
    use, intrinsic :: iso_c_binding, only : c_bool
    type(torch_tensor), intent(in) :: tensor       !! Tensor to compute gradients of
    logical, optional, intent(in)  :: retain_graph !! Should the computational graph be retained?

    ! Local arguments
    type(torch_tensor) :: external_gradient   !! External tensor used as an initial scaling of the gradient calculation
    logical(c_bool) :: retain_graph_value

    interface
      subroutine torch_tensor_backward_c(tensor_c, external_gradient_c, retain_graph_c) &
          bind(c, name = 'torch_tensor_backward')
        use, intrinsic :: iso_c_binding, only : c_bool, c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor_c
        type(c_ptr), value, intent(in) :: external_gradient_c
        logical(c_bool), value, intent(in) :: retain_graph_c
      end subroutine torch_tensor_backward_c
    end interface

    ! External gradient to provide to the back-propagation consisting of a tensor of ones
    ! TODO: Accept other external gradients as an optional argument
    call torch_tensor_ones(external_gradient, tensor%get_rank(), tensor%get_shape(), &
                           tensor%get_dtype(), tensor%get_device_type(), &
                           device_index=tensor%get_device_index())

    ! Do not retain the graph by default
    if (present(retain_graph)) then
      retain_graph_value = retain_graph
    else
      retain_graph_value = .false.
    end if

    ! Call back-propagation with the provided external gradient
    call torch_tensor_backward_c(tensor%p, external_gradient%p, retain_graph_value)

    ! Delete the external gradient tensor
    call torch_tensor_delete(external_gradient)
  end subroutine torch_tensor_backward

  !> Retrieves the gradient with respect to a Torch Tensor.
  subroutine torch_tensor_get_gradient(gradient, tensor)
    type(torch_tensor), intent(inout) :: gradient  !! Tensor holding the gradient
    type(torch_tensor), intent(in) :: tensor       !! Tensor to compute the gradient with respect to

    interface
      subroutine torch_tensor_get_gradient_c(tensor_c, gradient_c) &
          bind(c, name = 'torch_tensor_get_gradient')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: tensor_c
        type(c_ptr), value, intent(in) :: gradient_c
      end subroutine torch_tensor_get_gradient_c
    end interface

    if (.not. c_associated(gradient%p)) then
      write(*,*) "Error :: tensors for holding gradients must be constructed before retrieving values"
      stop 1
    end if
    call torch_tensor_get_gradient_c(tensor%p, gradient%p)
  end subroutine torch_tensor_get_gradient

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

  !> Deallocates a TorchScript model
  subroutine torch_model_delete(model)
    type(torch_model), intent(in) :: model  !! Torch Model to deallocate

    interface
      subroutine torch_jit_model_delete_c(model_c) &
          bind(c, name = 'torch_jit_module_delete')
        use, intrinsic :: iso_c_binding, only : c_ptr
        implicit none
        type(c_ptr), value, intent(in) :: model_c
      end subroutine torch_jit_model_delete_c
    end interface

    call torch_jit_model_delete_c(model%p)
  end subroutine torch_model_delete

end module ftorch
