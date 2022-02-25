module mod_torch

   use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_char, c_ptr, c_null_ptr
   implicit none

   type torch_module
      type(c_ptr) :: p = c_null_ptr
   end type torch_module

   type torch_tensor
      type(c_ptr) :: p = c_null_ptr
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

contains

   ! Torch Tensor API
   function torch_tensor_from_blob(data, ndims, shape, dtype, device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_ptr
      type(c_ptr), intent(in)        :: data
      integer(c_int), intent(in)     :: ndims
      integer(c_int64_t), intent(in) :: shape(*)
      integer(c_int), intent(in)     :: dtype
      integer(c_int), intent(in)     :: device
      type(torch_tensor)             :: tensor

      interface
         function torch_from_blob_c(data, ndims, shape, dtype, device) result(tensor) &
            bind(c, name='torch_from_blob')
            use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_ptr
            type(c_ptr), value, intent(in)    :: data
            integer(c_int), value, intent(in) :: ndims
            integer(c_int64_t), intent(in)    :: shape(*)
            integer(c_int), value, intent(in) :: dtype
            integer(c_int), value, intent(in) :: device
            type(c_ptr)                       :: tensor
         end function torch_from_blob_c
      end interface

      tensor%p = torch_from_blob_c(data, ndims, shape, dtype, device)
   end function torch_tensor_from_blob

   function torch_tensor_ones(ndims, shape, dtype, device) result(tensor)
      use, intrinsic :: iso_c_binding, only : c_int, c_int64_t
      integer(c_int), intent(in)     :: ndims
      integer(c_int64_t), intent(in) :: shape(*)
      integer(c_int), intent(in)     :: dtype
      integer(c_int), intent(in)     :: device
      type(torch_tensor)             :: tensor

      interface
         function torch_ones_c(ndims, shape, dtype, device) result(tensor) &
            bind(c, name='torch_ones')
            use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_ptr
            integer(c_int), value, intent(in) :: ndims
            integer(c_int64_t), intent(in)    :: shape(*)
            integer(c_int), value, intent(in) :: dtype
            integer(c_int), value, intent(in) :: device
            type(c_ptr)                       :: tensor
         end function torch_ones_c
      end interface

      tensor%p = torch_ones_c(ndims, shape, dtype, device)
   end function torch_tensor_ones

   subroutine torch_tensor_print(tensor)
      type(torch_tensor), intent(in) :: tensor

      interface
         subroutine torch_tensor_print_c(tensor) &
            bind(c, name='torch_tensor_print')
            use, intrinsic :: iso_c_binding, only : c_ptr
            type(c_ptr), value, intent(in) :: tensor
         end subroutine torch_tensor_print_c
      end interface

      call torch_tensor_print_c(tensor%p)
   end subroutine torch_tensor_print

   subroutine torch_tensor_delete(tensor)
      type(torch_tensor), intent(in) :: tensor

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
   function torch_module_load(filename) result(module)
      use, intrinsic :: iso_c_binding, only : c_char
      character(c_char), intent(in) :: filename(*)
      type(torch_module)            :: module

      interface
         function torch_jit_load_c(filename) result(module) &
            bind(c, name='torch_jit_load')
            use, intrinsic :: iso_c_binding, only : c_char, c_ptr
            character(c_char), intent(in) :: filename(*)
            type(c_ptr)                   :: module
         end function torch_jit_load_c
      end interface

      ! Need to append c_null_char at end of filename
      module%p = torch_jit_load_c(filename)
   end function torch_module_load

   function torch_module_forward(module, input_tensor) result(output_tensor)
      use, intrinsic :: iso_c_binding, only : c_ptr
      type(torch_module), intent(in) :: module
      type(torch_tensor), intent(in) :: input_tensor
      type(torch_tensor)             :: output_tensor

      interface
         function torch_jit_module_forward_c(module, input_tensor) result(output_tensor) &
            bind(c, name='torch_jit_module_forward')
            use, intrinsic :: iso_c_binding, only : c_ptr
            type(c_ptr), value, intent(in) :: module
            type(c_ptr), value, intent(in) :: input_tensor
            type(c_ptr)                    :: output_tensor
         end function torch_jit_module_forward_c
      end interface

      output_tensor%p = torch_jit_module_forward_c(module%p, input_tensor%p)
   end function torch_module_forward

   subroutine torch_module_delete(module)
      type(torch_module), intent(in) :: module

      interface
         subroutine torch_jit_module_delete_c(module) &
            bind(c, name='torch_jit_module_delete')
            use, intrinsic :: iso_c_binding, only : c_ptr
            type(c_ptr), value, intent(in) :: module
         end subroutine torch_jit_module_delete_c
      end interface

      call torch_jit_module_delete_c(module%p)
   end subroutine torch_module_delete

end module mod_torch
