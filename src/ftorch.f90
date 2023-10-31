module ftorch
   !! The ftorch module containing wrappers to access libtorch

   use, intrinsic :: iso_c_binding, only: c_int, c_int8_t, c_int16_t, c_int32_t, c_int64_t, c_int64_t, &
                                          c_float, c_double, c_char, c_ptr, c_null_ptr
   implicit none

   !> Type for holding a torch neural net (nn.Module).
   type torch_module
      type(c_ptr) :: p = c_null_ptr  !! pointer to the neural net module in memory
   end type torch_module

   !> Type for holding a torch tensor.
   type torch_tensor
      type(c_ptr) :: p = c_null_ptr  !! pointer to the tensor in memory
   end type torch_tensor

   ! From c_torch.h (torch_data_t)
   enum, bind(c)
      enumerator :: torch_kUInt8 = 0
      enumerator :: torch_kInt8 = 1
      enumerator :: torch_kInt16 = 2
      enumerator :: torch_kInt32 = 3
      enumerator :: torch_kInt64 = 4
      enumerator :: torch_kFloat16 = 5
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
      module procedure torch_tensor_from_array_c_float
      module procedure torch_tensor_from_array_c_double
      ! module procedure torch_tensor_from_array_c_int8_t
      ! module procedure torch_tensor_from_array_c_int16_t
      ! module procedure torch_tensor_from_array_c_int32_t
      ! module procedure torch_tensor_from_array_c_int64_t
   end interface

contains

   ! Torch Tensor API
   !> Exposes the given data as a tensor without taking ownership of the original data.
   !> This routine will take an (i, j, k) array and return an (k, j, i) tensor.
   function torch_tensor_from_blob(data, ndims, tensor_shape, dtype, device, layout) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_ptr

      ! Arguments
      type(c_ptr), intent(in)        :: data       !! Pointer to data
      integer(c_int), intent(in)     :: ndims      !! Number of dimensions of the tensor
      integer(c_int64_t), intent(in) :: tensor_shape(*)   !! Shape of the tensor
      integer(c_int), intent(in)     :: dtype      !! Data type of the tensor
      integer(c_int), intent(in)     :: device     !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)
      integer(c_int), intent(in)     :: layout(*)  !! Layout for strides for accessing data
      type(torch_tensor)             :: tensor     !! Returned tensor

      integer(c_int)                 :: i          !! loop index
      integer(c_int64_t)             :: strides(ndims) !! Strides for accessing data

      interface
         function torch_from_blob_c(data, ndims, tensor_shape, strides, dtype, device) result(tensor) &
            bind(c, name = 'torch_from_blob')
            use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_ptr
            type(c_ptr), value, intent(in)    :: data
            integer(c_int), value, intent(in) :: ndims
            integer(c_int64_t), intent(in)    :: tensor_shape(*)
            integer(c_int64_t), intent(in)    :: strides(*)
            integer(c_int), value, intent(in) :: dtype
            integer(c_int), value, intent(in) :: device
            type(c_ptr)                       :: tensor
         end function torch_from_blob_c
      end interface

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i - 1)) * tensor_shape(layout(i - 1))
      end do
      tensor%p = torch_from_blob_c(data, ndims, tensor_shape, strides, dtype, device)
   end function torch_tensor_from_blob

   !> This routine will take an (i, j, k) array and return an (k, j, i) tensor
   !> it is invoked from a set of interfaces `torch_tensor_from_array_dtype`
   function t_t_from_array(data_arr, tensor_shape, dtype, device) result(tensor)

     use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_double, c_loc

      ! Arguments
      type(c_ptr), intent(in)          :: data_arr       !! Pointer to data
      integer(c_int64_t), intent(in)   :: tensor_shape(:)   !! Shape of the tensor
      integer(c_int), intent(in)       :: dtype      !! Data type of the tensor
      integer(c_int), intent(in)       :: device     !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)
      type(torch_tensor)               :: tensor     !! Returned tensor

      integer(c_int)                   :: i          !! loop index
      integer(c_int64_t), allocatable  :: strides(:) !! Strides for accessing data
      integer(c_int), allocatable      :: layout(:)  !! Layout for strides for accessing data
      integer(c_int)                   :: ndims      !! Number of dimensions of the tensor

      interface
         function torch_from_blob_c(data, ndims, tensor_shape, strides, dtype, device) result(tensor) &
            bind(c, name = 'torch_from_blob')
            use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_ptr
            type(c_ptr), value, intent(in)    :: data
            integer(c_int), value, intent(in) :: ndims
            integer(c_int64_t), intent(in)    :: tensor_shape(*)
            integer(c_int64_t), intent(in)    :: strides(*)
            integer(c_int), value, intent(in) :: dtype
            integer(c_int), value, intent(in) :: device
            type(c_ptr)                       :: tensor
         end function torch_from_blob_c
      end interface

      ndims = size(tensor_shape)

      allocate(strides(ndims))
      allocate(layout(ndims))

      ! Fortran Layout
      do i = 1, ndims
          layout(i) = i
      end do

      strides(layout(1)) = 1
      do i = 2, ndims
        strides(layout(i)) = strides(layout(i - 1)) * tensor_shape(layout(i - 1))
      end do

      tensor%p = torch_from_blob_c(data_arr, ndims, tensor_shape, strides, dtype, device)

      deallocate(strides)
      deallocate(layout)

   end function t_t_from_array

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
            bind(c, name = 'torch_ones')
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
            bind(c, name = 'torch_zeros')
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
            bind(c, name = 'torch_tensor_print')
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
            bind(c, name = 'torch_tensor_delete')
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
            bind(c, name = 'torch_jit_load')
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
      integer(c_int) ::  n_inputs                     !! Number of tensors in `input_tensors`

      integer :: i
      type(c_ptr), dimension(n_inputs), target  :: input_ptrs

      interface
         subroutine torch_jit_module_forward_c(module, input_tensors, n_inputs, &
                                                        output_tensor) &
            bind(c, name = 'torch_jit_module_forward')
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
            bind(c, name = 'torch_jit_module_delete')
            use, intrinsic :: iso_c_binding, only : c_ptr
            type(c_ptr), value, intent(in) :: module
         end subroutine torch_jit_module_delete_c
      end interface

      call torch_jit_module_delete_c(module%p)
   end subroutine torch_module_delete

   ! Series of interface functions
   function torch_tensor_from_array_c_double(data_arr, tensor_shape, device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_double, c_loc
      real(c_double), intent(in), target :: data_arr(*)   !! Fortran array of data
      ! real(c_double), intent(in), target :: data_arr(*)   !! Fortran array of data
      integer(c_int64_t), intent(in)   :: tensor_shape(:)   !! Shape of the tensor
      integer(c_int), parameter :: dtype = torch_kFloat64
      integer(c_int), intent(in)       :: device     !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)
      type(torch_tensor)               :: tensor     !! Returned tensor

      tensor = t_t_from_array(c_loc(data_arr), tensor_shape, dtype, device)

   end function torch_tensor_from_array_c_double

   function torch_tensor_from_array_c_float(data_arr, tensor_shape, device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_float, c_loc
      real(c_float), intent(in), target :: data_arr(*)   !! Fortran array of data
      integer(c_int64_t), intent(in)   :: tensor_shape(:)   !! Shape of the tensor
      integer(c_int), parameter :: dtype = torch_kFloat32
      integer(c_int), intent(in)       :: device     !! Device on which the tensor will live on (torch_kCPU or torch_kGPU)
      type(torch_tensor)               :: tensor     !! Returned tensor

     tensor = t_t_from_array(c_loc(data_arr), tensor_shape, dtype, device)

   end function torch_tensor_from_array_c_float

end module ftorch
