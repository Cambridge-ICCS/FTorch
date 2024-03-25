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

   ! MPI configuration
   integer :: rank, ierr, i

   call mpi_init(ierr)
   call mpi_comm_rank(mpi_comm_world, rank, ierr)

   ! Get TorchScript model file as a command line argument
   num_args = command_argument_count()
   allocate(args(num_args))
   do ix = 1, num_args
       call get_command_argument(ix,args(ix))
   end do

   ! Initialise data
   in_data = [(rank + i, i=0,4)]

   ! Create Torch input/output tensors from the above arrays
   in_tensor(1) = torch_tensor_from_array(in_data, tensor_layout, torch_kCUDA, device_index=rank)
   out_tensor = torch_tensor_from_array(out_data, tensor_layout, torch_kCUDA, device_index=rank)

   ! Load ML model
   model = torch_module_load(args(1), device_type=torch_kCUDA, device_index=rank)

   ! Infer
   call torch_module_forward(model, in_tensor, n_inputs, out_tensor)
   write (*,*) rank, ":", out_data(:)

   ! Cleanup
   call torch_module_delete(model)
   call torch_tensor_delete(in_tensor(1))
   call torch_tensor_delete(out_tensor)

end program inference
