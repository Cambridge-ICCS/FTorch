program inference

   ! Import precision info from iso
   use, intrinsic :: iso_fortran_env, only : sp => real32

   ! Import our library for interfacing with PyTorch
   use ftorch

   ! Import MPI
   use mpi

   implicit none
  
   ! Set precision for reals
   integer, parameter :: wp = sp
   
   integer :: num_args, ix
   character(len=128), dimension(:), allocatable :: args

   ! Set up Fortran data structures
   real(wp), dimension(5), target :: in_data
   real(wp), dimension(5), target :: out_data
   integer, parameter :: n_inputs = 1
   integer :: tensor_layout(1) = [1]

   ! Set up Torch data structures
   type(torch_module) :: model
   type(torch_tensor), dimension(1) :: in_tensor
   type(torch_tensor) :: out_tensor

   ! TODO: MPI setup

   ! Get TorchScript model file as a command line argument
   num_args = command_argument_count()
   allocate(args(num_args))
   do ix = 1, num_args
       call get_command_argument(ix,args(ix))
   end do

   ! Initialise data
   ! TODO: Different inputs for different ranks
   in_data = [0.0, 1.0, 2.0, 3.0, 4.0]

   ! Create Torch input/output tensors from the above arrays
   ! TODO: Use GPU
   in_tensor(1) = torch_tensor_from_array(in_data, tensor_layout, torch_kCPU)
   out_tensor = torch_tensor_from_array(out_data, tensor_layout, torch_kCPU)

   ! Load ML model
   ! TODO: Use GPU
   model = torch_module_load(args(1))

   ! Infer
   call torch_module_forward(model, in_tensor, n_inputs, out_tensor)
   ! TODO: Write rank, too
   write (*,*) out_data(:)

   ! Cleanup
   call torch_module_delete(model)
   call torch_tensor_delete(in_tensor(1))
   call torch_tensor_delete(out_tensor)

end program inference
