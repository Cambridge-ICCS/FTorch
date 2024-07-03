program inference

   ! Import precision info from iso
   use, intrinsic :: iso_fortran_env, only : sp => real32

   ! Import our library for interfacing with PyTorch
   use ftorch

   implicit none
  
   ! Set working precision for reals
   integer, parameter :: wp = sp
   
   integer :: num_args, ix
   character(len=128), dimension(:), allocatable :: args

   ! Set up Fortran data structures
   real(wp), dimension(4), target :: in_data1
   real(wp), dimension(4), target :: in_data2
   real(wp), dimension(4), target :: out_data1
   real(wp), dimension(4), target :: out_data2
   integer :: tensor_layout(1) = [1]

   ! Set up Torch data structures
   ! The net, a vector of input tensors (in this case we only have one), and the
   ! output tensor
   type(torch_model) :: model
   type(torch_tensor), dimension(2) :: in_tensors
   type(torch_tensor), dimension(2) :: out_tensors

   ! Get TorchScript model file as a command line argument
   num_args = command_argument_count()
   allocate(args(num_args))
   do ix = 1, num_args
       call get_command_argument(ix,args(ix))
   end do

   ! Initialise data
   in_data1(:) = [0.0, 1.0, 2.0, 3.0]
   in_data2(:) = [0.0, -1.0, -2.0, -3.0]

   ! Create Torch input/output tensors from the above arrays
   call torch_tensor_from_array(in_tensors(1), in_data1, tensor_layout,        &
                                device_type=torch_kCPU)
   call torch_tensor_from_array(in_tensors(2), in_data2, tensor_layout,        &
                                device_type=torch_kCPU)
   call torch_tensor_from_array(out_tensors(1), out_data1, tensor_layout,      &
                                device_type=torch_kCPU)
   call torch_tensor_from_array(out_tensors(2), out_data2, tensor_layout,      &
                                device_type=torch_kCPU)

   ! Load ML model
   call torch_model_load(model, args(1))

   ! Infer
   call torch_model_forward(model, in_tensors, out_tensors)
   write (*,*) out_data1
   write (*,*) out_data2

   ! Cleanup
   call torch_model_delete(model)
   call torch_tensor_delete(in_tensors(1))
   call torch_tensor_delete(in_tensors(2))
   call torch_tensor_delete(out_tensors(1))
   call torch_tensor_delete(out_tensors(2))

end program inference
