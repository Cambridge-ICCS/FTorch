module mod_torch

   use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_char, c_ptr
   implicit none

   public :: torch_ones, torch_from_blob, torch_tensor_print, torch_tensor_delete, torch_jit_load, torch_jit_module_forward

   interface

      function torch_from_blob(data, dims, shape, dtype, device) result(tensor) &
         bind(c, name='torch_from_blob')
         import :: c_int, c_int64_t, c_ptr
         type(c_ptr), value, intent(in) :: data
         integer(c_int), value, intent(in) :: dims
         integer(c_int64_t), intent(in) :: shape(*)
         integer(c_int), value, intent(in) :: dtype
         integer(c_int), value, intent(in) :: device
         type(c_ptr) :: tensor
      end function

      function torch_ones(ndims, shape, dtype, device) result(tensor) &
         bind(c, name='torch_ones')
         import :: c_int, c_int64_t, c_ptr
         integer(c_int), value, intent(in) :: ndims
         integer(c_int64_t), intent(in) :: shape(*)
         integer(c_int), value, intent(in) :: dtype
         integer(c_int), value, intent(in) :: device
         type(c_ptr) :: tensor
      end function

      subroutine torch_tensor_print(tensor) &
         bind(c, name='torch_tensor_print')
         import :: c_ptr
         type(c_ptr), value, intent(in) :: tensor
      end subroutine torch_tensor_print

      subroutine torch_tensor_delete(tensor) &
         bind(c, name='torch_tensor_delete')
         import :: c_ptr
         type(c_ptr), value, intent(in) :: tensor
      end subroutine torch_tensor_delete

      ! Need to append c_null_char at end of filename
      function torch_jit_load(filename) result(module) &
         bind(c, name='torch_jit_load')
         import :: c_char, c_ptr
         character(c_char), intent(in) :: filename(*)
         type(c_ptr) :: module
      end function torch_jit_load

      function torch_jit_module_forward(module, input_tensor) result(output_tensor) &
         bind(c, name='torch_jit_module_forward')
         import :: c_ptr
         type(c_ptr), value, intent(in) :: module
         type(c_ptr), value, intent(in) :: input_tensor
         type(c_ptr) :: output_tensor
      end function torch_jit_module_forward

      subroutine torch_jit_module_delete(module) &
         bind(c, name='torch_jit_module_delete')
         import :: c_ptr
         type(c_ptr), value, intent(in) :: module
      end subroutine torch_jit_module_delete
   end interface

end module mod_torch
