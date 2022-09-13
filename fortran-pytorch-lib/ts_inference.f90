program inference

   ! Imports primitives used to interface with C
   use, intrinsic :: iso_c_binding, only: c_int64_t, c_float, c_char, c_null_char, c_ptr, c_loc
   ! Import our library for interfacing with PyTorch
   use ftorch

   implicit none

   ! Set up types of input and output data and the interface with C
   type(torch_module) :: model
   type(torch_tensor) :: input_tensor, output_tensor

   real(c_float), allocatable, target :: input_data(:)
   real(c_float), dimension(:), pointer :: output_data

   type(c_ptr) :: input_data_ptr, output_data_ptr

   integer(c_int), parameter :: input_dims = 4
   integer(c_int64_t) :: input_shape(input_dims) = [1, 3, 224, 224]

   character(len=:), allocatable :: filename

   ! Allocate one-dimensional input_data array, based on multiplication of all input_dimension sizes
   allocate(input_data(product(input_shape)))

   ! Initialise every element to 1
   input_data = 1.0d0

   ! Get pointer to the input data (like doing &input_data in C)
   input_data_ptr = c_loc(input_data)

   ! Create input tensor from the above array
   input_tensor = torch_tensor_from_blob(input_data_ptr, input_dims, input_shape, torch_kFloat32, torch_kCPU)

   ! Load ML model (edit this line to use different models)
   model = torch_module_load(c_char_"annotated_cpu.pt"//c_null_char)

   ! Deploy
   output_tensor = torch_module_forward(model, input_tensor)

   ! Cleanup
   call torch_module_delete(model)
   call torch_tensor_delete(input_tensor)
   call torch_tensor_delete(output_tensor)
   deallocate(input_data)

end program inference
