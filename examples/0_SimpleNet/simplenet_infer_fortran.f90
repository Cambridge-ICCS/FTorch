program inference

   ! Imports primitives used to interface with C
   use, intrinsic :: iso_c_binding, only: c_int64_t, c_float, c_char, c_null_char, c_ptr, c_loc
   ! Import our library for interfacing with PyTorch
   use ftorch

   implicit none

   integer :: num_args, ix
   character(len=128), dimension(:), allocatable :: args

   ! Set up types of input and output data and the interface with C
   type(torch_module) :: model
   type(torch_tensor), dimension(1) :: in_tensor
   type(torch_tensor) :: out_tensor

   real(c_float), dimension(:), allocatable, target :: in_data
   integer(c_int), parameter :: n_inputs = 1
   real(c_float), dimension(:), allocatable, target :: out_data

   integer(c_int), parameter :: tensor_dims = 1
   integer(c_int64_t) :: tensor_shape(tensor_dims) = [5]
   integer(c_int) :: tensor_layout(tensor_dims) = [1]

   ! Get TorchScript model file as a command line argument
   num_args = command_argument_count()
   allocate(args(num_args))
   do ix = 1, num_args
       call get_command_argument(ix,args(ix))
   end do

   ! Allocate one-dimensional input/output arrays, based on multiplication of all input/output dimension sizes
   allocate(in_data(tensor_shape(1)))
   allocate(out_data(tensor_shape(1)))

   ! Initialise data
   in_data = 1.0d0

   ! Create input/output tensors from the above arrays
   in_tensor(1) = torch_tensor_from_blob(c_loc(in_data), tensor_dims, tensor_shape, torch_kFloat32, torch_kCPU, tensor_layout)
   out_tensor = torch_tensor_from_blob(c_loc(out_data), tensor_dims, tensor_shape, torch_kFloat32, torch_kCPU, tensor_layout)

   ! Load ML model (edit this line to use different models)
   model = torch_module_load(trim(args(1))//c_null_char)

   ! Infer
   call torch_module_forward(model, in_tensor, n_inputs, out_tensor)
   write (*,*) out_data(:)

   ! Cleanup
   call torch_module_delete(model)
   call torch_tensor_delete(in_tensor(1))
   call torch_tensor_delete(out_tensor)
   deallocate(in_data)
   deallocate(out_data)

end program inference
