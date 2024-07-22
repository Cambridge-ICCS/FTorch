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
   real(wp), dimension(5), target :: in_data
   real(wp), dimension(5), target :: dummy_data
   real(wp), dimension(:), pointer :: out_data
   integer :: tensor_layout(1) = [1]

   ! Set up Torch data structures
   ! The net, a vector of input tensors (in this case we only have one), and the output tensor
   type(torch_model) :: model
   type(torch_tensor), dimension(1) :: in_tensors
   type(torch_tensor), dimension(1) :: out_tensors

   ! Get TorchScript model file as a command line argument
   num_args = command_argument_count()
   allocate(args(num_args))
   do ix = 1, num_args
       call get_command_argument(ix,args(ix))
   end do

   ! Initialise data
   in_data = [0.0, 1.0, 2.0, 3.0, 4.0]

   ! Create Torch input tensor from the above array
   call torch_tensor_from_array(in_tensors(1), in_data, tensor_layout, torch_kCPU)

   ! Create an empty Torch output tensor
   ! TODO: Drop dummy_data and initialise out_tensors with torch_tensor_empty
   call torch_tensor_from_array(out_tensors(1), dummy_data, tensor_layout, torch_kCPU)

   ! Load ML model
   call torch_model_load(model, args(1))

   ! Infer
   call torch_model_forward(model, in_tensors, out_tensors)

   ! Extract the output as a Fortran array
   allocate(out_data(5))
   call torch_tensor_to_array(out_tensors(1), out_data)
   write (*,*) out_data(:)

   ! Cleanup
   nullify(out_data)
   call torch_delete(model)
   call torch_delete(in_tensors)
   call torch_delete(out_tensors)

end program inference
