module ftorch

  use, intrinsic :: iso_c_binding, only: c_int, c_int8_t, c_int16_t, c_int32_t, c_int64_t, c_int64_t, &
    c_float, c_double, c_char, c_ptr, c_null_ptr
  use, intrinsic :: iso_fortran_env, only: int8, int16, int32, int64, real32, real64
  implicit none

  type torch_module
    type(c_ptr) :: p = c_null_ptr
  end type torch_module

  type torch_tensor
    type(c_ptr) :: p = c_null_ptr
  end type torch_tensor

  ! From c_torch.h (torch_data_t)
  enum, bind(c)
    enumerator :: torch_kUInt8 = 0 ! not supported in fortran
    enumerator :: torch_kInt8 = 1
    enumerator :: torch_kInt16 = 2
    enumerator :: torch_kInt32 = 3
    enumerator :: torch_kInt64 = 4
    enumerator :: torch_kFloat16 = 5 ! not supported in fortran
    enumerator :: torch_kFloat32 = 6
    enumerator :: torch_kFloat64 = 7
  end enum


  ! From c_torch.h (torch_device_t)
  enum, bind(c)
    enumerator :: torch_kCPU = 0
    enumerator :: torch_kCUDA = 1
  end enum

  ! Interface for calculating tensor from array for different possible input types
  interface torch_tensor_from_array
    module procedure torch_tensor_from_array_int8_1
    module procedure torch_tensor_from_array_int8_2
    module procedure torch_tensor_from_array_int8_3
    module procedure torch_tensor_from_array_int8_4
    module procedure torch_tensor_from_array_int16_1
    module procedure torch_tensor_from_array_int16_2
    module procedure torch_tensor_from_array_int16_3
    module procedure torch_tensor_from_array_int16_4
    module procedure torch_tensor_from_array_int32_1
    module procedure torch_tensor_from_array_int32_2
    module procedure torch_tensor_from_array_int32_3
    module procedure torch_tensor_from_array_int32_4
    module procedure torch_tensor_from_array_int64_1
    module procedure torch_tensor_from_array_int64_2
    module procedure torch_tensor_from_array_int64_3
    module procedure torch_tensor_from_array_int64_4
    module procedure torch_tensor_from_array_real32_1
    module procedure torch_tensor_from_array_real32_2
    module procedure torch_tensor_from_array_real32_3
    module procedure torch_tensor_from_array_real32_4
    module procedure torch_tensor_from_array_real64_1
    module procedure torch_tensor_from_array_real64_2
    module procedure torch_tensor_from_array_real64_3
    module procedure torch_tensor_from_array_real64_4
  end interface

  interface
    function torch_from_blob_c(data, ndims, tensor_shape, strides, dtype, device) result(tensor_p) &
        bind(c, name='torch_from_blob')
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_ptr
      type(c_ptr), value, intent(in)    :: data
      integer(c_int), value, intent(in) :: ndims
      integer(c_int64_t), intent(in)    :: tensor_shape(*)
      integer(c_int64_t), intent(in)    :: strides(*)
      integer(c_int), value, intent(in) :: dtype
      integer(c_int), value, intent(in) :: device
      type(c_ptr)                       :: tensor_p
    end function torch_from_blob_c
  end interface

contains

  ! Torch Tensor API
  !> Exposes the given data as a tensor without taking ownership of the original data.
  !> This routine will take an (i, j, k) array and return an (k, j, i) tensor.
  function torch_tensor_from_blob(data, ndims, tensor_shape, dtype, device, layout) result(tensor)
    use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_ptr
    type(c_ptr), intent(in)        :: data       !! Pointer to data
    integer(c_int), intent(in)     :: ndims      !! Number of dimensions of the tensor
    integer(c_int64_t), intent(in) :: tensor_shape(*)   !! Shape of the tensor
    integer(c_int), intent(in)     :: dtype      !! Data type of the tensor
    integer(c_int), intent(in)     :: device     !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)
    integer(c_int), intent(in)     :: layout(*)  !! Layout for strides for accessing data
    type(torch_tensor)             :: tensor     !! Returned tensor

    integer(c_int)                 :: i          !! loop index
    integer(c_int64_t)             :: strides(ndims) !! Strides for accessing data

    strides(layout(1)) = 1
    do i = 2, ndims
      strides(layout(i)) = strides(layout(i-1)) * tensor_shape(layout(i-1))
    end do
    tensor%p = torch_from_blob_c(data, ndims, tensor_shape, strides, dtype, device)
  end function torch_tensor_from_blob

  !> Returns a tensor filled with the scalar value 1.
  function torch_tensor_ones(ndims, tensor_shape, dtype, device) result(tensor)
    use, intrinsic :: iso_c_binding, only : c_int, c_int64_t
    integer(c_int), intent(in)     :: ndims      !! Number of dimensions of the tensor
    integer(c_int64_t), intent(in) :: tensor_shape(*)   !! Shape of the tensor
    integer(c_int), intent(in)     :: dtype      !! Data type of the tensor
    integer(c_int), intent(in)     :: device     !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)
    type(torch_tensor)             :: tensor     !! Returned tensor

    interface
      function torch_ones_c(ndims, tensor_shape, dtype, device) result(tensor) &
          bind(c, name='torch_ones')
        use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_ptr
        integer(c_int), value, intent(in) :: ndims
        integer(c_int64_t), intent(in)    :: tensor_shape(*)
        integer(c_int), value, intent(in) :: dtype
        integer(c_int), value, intent(in) :: device
        type(c_ptr)                       :: tensor
      end function torch_ones_c
    end interface

    tensor%p = torch_ones_c(ndims, tensor_shape, dtype, device)
  end function torch_tensor_ones

  !> Returns a tensor filled with the scalar value 0.
  function torch_tensor_zeros(ndims, tensor_shape, dtype, device) result(tensor)
    use, intrinsic :: iso_c_binding, only : c_int, c_int64_t
    integer(c_int), intent(in)     :: ndims      !! Number of dimensions of the tensor
    integer(c_int64_t), intent(in) :: tensor_shape(*)   !! Shape of the tensor
    integer(c_int), intent(in)     :: dtype      !! Data type of the tensor
    integer(c_int), intent(in)     :: device     !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)
    type(torch_tensor)             :: tensor     !! Returned tensor

    interface
      function torch_zeros_c(ndims, tensor_shape, dtype, device) result(tensor) &
          bind(c, name='torch_zeros')
        use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_ptr
        integer(c_int), value, intent(in) :: ndims
        integer(c_int64_t), intent(in)    :: tensor_shape(*)
        integer(c_int), value, intent(in) :: dtype
        integer(c_int), value, intent(in) :: device
        type(c_ptr)                       :: tensor
      end function torch_zeros_c
    end interface

    tensor%p = torch_zeros_c(ndims, tensor_shape, dtype, device)
  end function torch_tensor_zeros

  !> Prints the contents of a tensor.
  subroutine torch_tensor_print(tensor)
    type(torch_tensor), intent(in) :: tensor     !! Input tensor

    interface
      subroutine torch_tensor_print_c(tensor) &
          bind(c, name='torch_tensor_print')
        use, intrinsic :: iso_c_binding, only : c_ptr
        type(c_ptr), value, intent(in) :: tensor
      end subroutine torch_tensor_print_c
    end interface

    call torch_tensor_print_c(tensor%p)
  end subroutine torch_tensor_print

  !> Deallocates a tensor.
  subroutine torch_tensor_delete(tensor)
    type(torch_tensor), intent(in) :: tensor     !! Input tensor

    interface
      subroutine torch_tensor_delete_c(tensor) &
          bind(c, name='torch_tensor_delete')
        use, intrinsic :: iso_c_binding, only : c_ptr
        type(c_ptr), value, intent(in) :: tensor
      end subroutine torch_tensor_delete_c
    end interface

    call torch_tensor_delete_c(tensor%p)
  end subroutine torch_tensor_delete

  ! Torch Module API
  !> Loads a Torch Script module (pre-trained PyTorch model saved with Torch Script)
  function torch_module_load(filename) result(module)
    use, intrinsic :: iso_c_binding, only : c_null_char
    character(*), intent(in) :: filename !! Filename of Torch Script module
    type(torch_module)            :: module      !! Returned deserialized module

    interface
      function torch_jit_load_c(filename) result(module) &
          bind(c, name='torch_jit_load')
        use, intrinsic :: iso_c_binding, only : c_char, c_ptr
        character(c_char), intent(in) :: filename(*)
        type(c_ptr)                   :: module
      end function torch_jit_load_c
    end interface

    ! Need to append c_null_char at end of filename
    module%p = torch_jit_load_c(trim(adjustl(filename))//c_null_char)
    end function torch_module_load

    !> Performs a forward pass of the module with the input tensors
    subroutine torch_module_forward(module, input_tensors, n_inputs, output_tensor)
      use, intrinsic :: iso_c_binding, only : c_ptr, c_int, c_loc
      type(torch_module), intent(in) :: module        !! Module
      type(torch_tensor), intent(in), dimension(:) :: input_tensors  !! Array of Input tensors
      type(torch_tensor), intent(in) :: output_tensor !! Returned output tensors
      integer(c_int) ::  n_inputs

      integer :: i
      type(c_ptr), dimension(n_inputs), target  :: input_ptrs

      interface
        subroutine torch_jit_module_forward_c(module, input_tensors, n_inputs, &
            output_tensor) &
            bind(c, name='torch_jit_module_forward')
          use, intrinsic :: iso_c_binding, only : c_ptr, c_int
          type(c_ptr), value, intent(in) :: module
          type(c_ptr), value, intent(in) :: input_tensors
          integer(c_int), value, intent(in) :: n_inputs
          type(c_ptr), value, intent(in) :: output_tensor
        end subroutine torch_jit_module_forward_c
      end interface

      ! Assign array of pointers to the input tensors
      do i = 1, n_inputs
        input_ptrs(i) = input_tensors(i)%p
      end do

      call torch_jit_module_forward_c(module%p, c_loc(input_ptrs), n_inputs, output_tensor%p)
    end subroutine torch_module_forward

    !> Deallocates a Torch Script module
    subroutine torch_module_delete(module)
      type(torch_module), intent(in) :: module     !! Module

      interface
        subroutine torch_jit_module_delete_c(module) &
            bind(c, name='torch_jit_module_delete')
          use, intrinsic :: iso_c_binding, only : c_ptr
          type(c_ptr), value, intent(in) :: module
        end subroutine torch_jit_module_delete_c
      end interface

      call torch_jit_module_delete_c(module%p)
    end subroutine torch_module_delete

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_int8_1(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : int8

      ! inputs
      integer(kind=int8), intent(in), target :: data_in(:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(1) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(1)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kInt8 !! data type
      integer(c_int64_t)        :: strides(1)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 1                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_int8_1

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_int8_2(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : int8

      ! inputs
      integer(kind=int8), intent(in), target :: data_in(:,:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(2) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(2)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kInt8 !! data type
      integer(c_int64_t)        :: strides(2)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 2                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_int8_2

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_int8_3(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : int8

      ! inputs
      integer(kind=int8), intent(in), target :: data_in(:,:,:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(3) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(3)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kInt8 !! data type
      integer(c_int64_t)        :: strides(3)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 3                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_int8_3

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_int8_4(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : int8

      ! inputs
      integer(kind=int8), intent(in), target :: data_in(:,:,:,:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(4) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(4)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kInt8 !! data type
      integer(c_int64_t)        :: strides(4)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 4                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_int8_4

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_int16_1(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : int16

      ! inputs
      integer(kind=int16), intent(in), target :: data_in(:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(1) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(1)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kInt16 !! data type
      integer(c_int64_t)        :: strides(1)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 1                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_int16_1

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_int16_2(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : int16

      ! inputs
      integer(kind=int16), intent(in), target :: data_in(:,:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(2) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(2)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kInt16 !! data type
      integer(c_int64_t)        :: strides(2)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 2                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_int16_2

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_int16_3(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : int16

      ! inputs
      integer(kind=int16), intent(in), target :: data_in(:,:,:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(3) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(3)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kInt16 !! data type
      integer(c_int64_t)        :: strides(3)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 3                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_int16_3

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_int16_4(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : int16

      ! inputs
      integer(kind=int16), intent(in), target :: data_in(:,:,:,:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(4) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(4)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kInt16 !! data type
      integer(c_int64_t)        :: strides(4)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 4                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_int16_4

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_int32_1(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : int32

      ! inputs
      integer(kind=int32), intent(in), target :: data_in(:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(1) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(1)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kInt32 !! data type
      integer(c_int64_t)        :: strides(1)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 1                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_int32_1

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_int32_2(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : int32

      ! inputs
      integer(kind=int32), intent(in), target :: data_in(:,:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(2) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(2)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kInt32 !! data type
      integer(c_int64_t)        :: strides(2)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 2                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_int32_2

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_int32_3(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : int32

      ! inputs
      integer(kind=int32), intent(in), target :: data_in(:,:,:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(3) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(3)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kInt32 !! data type
      integer(c_int64_t)        :: strides(3)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 3                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_int32_3

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_int32_4(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : int32

      ! inputs
      integer(kind=int32), intent(in), target :: data_in(:,:,:,:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(4) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(4)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kInt32 !! data type
      integer(c_int64_t)        :: strides(4)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 4                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_int32_4

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_int64_1(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : int64

      ! inputs
      integer(kind=int64), intent(in), target :: data_in(:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(1) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(1)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kInt64 !! data type
      integer(c_int64_t)        :: strides(1)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 1                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_int64_1

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_int64_2(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : int64

      ! inputs
      integer(kind=int64), intent(in), target :: data_in(:,:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(2) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(2)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kInt64 !! data type
      integer(c_int64_t)        :: strides(2)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 2                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_int64_2

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_int64_3(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : int64

      ! inputs
      integer(kind=int64), intent(in), target :: data_in(:,:,:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(3) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(3)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kInt64 !! data type
      integer(c_int64_t)        :: strides(3)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 3                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_int64_3

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_int64_4(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : int64

      ! inputs
      integer(kind=int64), intent(in), target :: data_in(:,:,:,:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(4) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(4)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kInt64 !! data type
      integer(c_int64_t)        :: strides(4)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 4                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_int64_4

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_real32_1(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : real32

      ! inputs
      real(kind=real32), intent(in), target :: data_in(:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(1) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(1)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kFloat32 !! data type
      integer(c_int64_t)        :: strides(1)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 1                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_real32_1

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_real32_2(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : real32

      ! inputs
      real(kind=real32), intent(in), target :: data_in(:,:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(2) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(2)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kFloat32 !! data type
      integer(c_int64_t)        :: strides(2)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 2                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_real32_2

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_real32_3(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : real32

      ! inputs
      real(kind=real32), intent(in), target :: data_in(:,:,:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(3) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(3)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kFloat32 !! data type
      integer(c_int64_t)        :: strides(3)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 3                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_real32_3

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_real32_4(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : real32

      ! inputs
      real(kind=real32), intent(in), target :: data_in(:,:,:,:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(4) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(4)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kFloat32 !! data type
      integer(c_int64_t)        :: strides(4)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 4                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_real32_4

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_real64_1(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : real64

      ! inputs
      real(kind=real64), intent(in), target :: data_in(:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(1) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(1)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kFloat64 !! data type
      integer(c_int64_t)        :: strides(1)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 1                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_real64_1

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_real64_2(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : real64

      ! inputs
      real(kind=real64), intent(in), target :: data_in(:,:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(2) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(2)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kFloat64 !! data type
      integer(c_int64_t)        :: strides(2)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 2                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_real64_2

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_real64_3(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : real64

      ! inputs
      real(kind=real64), intent(in), target :: data_in(:,:,:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(3) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(3)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kFloat64 !! data type
      integer(c_int64_t)        :: strides(3)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 3                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_real64_3

    !> return a torch tensor pointing to data_in array
    function torch_tensor_from_array_real64_4(data_in, layout, c_device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      use, intrinsic :: iso_fortran_env, only : real64

      ! inputs
      real(kind=real64), intent(in), target :: data_in(:,:,:,:)   !! input data that tensor will point at
      integer, intent(in)        :: layout(4) !! control order of indices
      integer(c_int), intent(in) :: c_device         !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)

      ! output tensory
      type(torch_tensor) :: tensor     !! Returned tensor

      ! local data
      integer(c_int64_t)        :: c_tensor_shape(4)           !! Shape of the tensor
      integer(c_int), parameter :: c_dtype = torch_kFloat64 !! data type
      integer(c_int64_t)        :: strides(4)                  !! Strides for accessing data
      integer(c_int), parameter :: ndims = 4                   !! number of dimension of input data
      integer                   :: i

      c_tensor_shape = shape(data_in)

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i-1)) * c_tensor_shape(layout(i-1))
      end do

      tensor%p = torch_from_blob_c(c_loc(data_in), ndims, c_tensor_shape, strides, c_dtype, c_device)

    end function torch_tensor_from_array_real64_4


  end module ftorch
