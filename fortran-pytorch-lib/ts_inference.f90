program inference

   use, intrinsic :: iso_c_binding, only: c_int64_t, c_float, c_char, c_null_char, c_ptr, c_loc
   use mod_torch

   implicit none

   type(torch_module) :: model
   type(torch_tensor) :: input_tensor, output_tensor
   real(c_float), allocatable, target :: input_data(:)
   real(c_float), dimension(:), pointer :: output_data
   type(c_ptr) :: input_data_ptr, output_data_ptr
   integer(c_int), parameter :: input_dims = 4
   integer(c_int64_t) :: input_shape(input_dims) = [1, 3, 224, 224]
   character(len=:), allocatable :: filename

   allocate(input_data(product(input_shape)))
   input_data = 1.0d0
   input_data_ptr = c_loc(input_data)

   ! Create input tensor
   input_tensor = torch_tensor_from_blob(input_data_ptr, input_dims, input_shape, torch_kFloat32, torch_kCPU)
   ! Load ML model
   model = torch_module_load(c_char_"../annotated_cpu.pt"//c_null_char)
   ! Deploy
   output_tensor = torch_module_forward(model, input_tensor)

   ! Cleanup
   call torch_module_delete(model)
   call torch_tensor_delete(input_tensor)
   call torch_tensor_delete(output_tensor)
   deallocate(input_data)

end program inference
