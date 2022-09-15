program inference

   ! Imports primitives used to interface with C
   use, intrinsic :: iso_c_binding, only: c_int64_t, c_float, c_char, c_null_char, c_ptr, c_loc
   ! Import our library for interfacing with PyTorch
   use ftorch

   implicit none

   ! Set up types of input and output data and the interface with C
   type(torch_module) :: model
   type(torch_tensor) :: in_tensor, out_tensor

   real(c_float), dimension(:,:,:,:), allocatable, target :: in_data
   real(c_float), dimension(:,:), allocatable, target :: out_data

   integer(c_int), parameter :: in_dims = 4
   integer(c_int64_t) :: in_shape(in_dims) = [1, 3, 224, 224]
   integer(c_int), parameter :: out_dims = 2
   integer(c_int64_t) :: out_shape(out_dims) = [1, 1000]

   character(len=:), allocatable :: filename

   ! Allocate one-dimensional input/output arrays, based on multiplication of all input/output dimension sizes
   allocate(in_data(in_shape(1), in_shape(2), in_shape(3), in_shape(4)))
   allocate(out_data(out_shape(1), out_shape(2)))

   ! Initialise data
   in_data = 1.0d0

   ! Create input/output tensors from the above arrays
   in_tensor = torch_tensor_from_blob(c_loc(in_data), in_dims, in_shape, torch_kFloat32, torch_kCPU)
   out_tensor = torch_tensor_from_blob(c_loc(out_data), out_dims, out_shape, torch_kFloat32, torch_kCPU)

   ! Load ML model (edit this line to use different models)
   model = torch_module_load(c_char_"scripted_cpu.pt"//c_null_char)

   ! Infer
   call torch_module_forward(model, in_tensor, out_tensor)
   write (*,*) out_data(1, 1000)

   ! Cleanup
   call torch_module_delete(model)
   call torch_tensor_delete(in_tensor)
   call torch_tensor_delete(out_tensor)
   deallocate(in_data)
   deallocate(out_data)

end program inference
