program inference

   ! Import precision info from iso
   use, intrinsic :: iso_fortran_env, only : sp => real32

   ! Import our library for interfacing with PyTorch
   use ftorch, only : torch_model, torch_tensor, torch_kCPU, torch_delete, &
                      torch_tensor_from_array, torch_model_load, torch_model_forward

   ! Import our tools module for testing utils
   use ftorch_test_utils, only : assert_allclose

   ! Import MPI
   use mpi, only : mpi_init, mpi_finalize, mpi_comm_world, mpi_comm_rank

   implicit none

   ! Set working precision for reals
   integer, parameter :: wp = sp

   integer :: num_args, ix
   character(len=128), dimension(:), allocatable :: args

   ! Set up Fortran data structures
   real(wp), dimension(5), target :: in_data
   real(wp), dimension(5), target :: out_data
   real(wp), dimension(5), target :: expected
   integer, parameter :: tensor_layout(1) = [1]

   ! Set up Torch data structures
   ! The net, a vector of input tensors (in this case we only have one), and the output tensor
   type(torch_model) :: model
   type(torch_tensor), dimension(1) :: in_tensors
   type(torch_tensor), dimension(1) :: out_tensors

   ! Flag for testing
   logical :: test_pass

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

   ! Initialise data and print the values used on each MPI rank
   in_data = [(rank + i, i = 0, 4)]
   write (6, "('input on rank ',i1,': [',4(f5.1,','),f5.1,']')") rank, in_data(:)

   ! Create Torch input/output tensors from the above arrays
   call torch_tensor_from_array(in_tensors(1), in_data, tensor_layout, torch_kCPU)
   call torch_tensor_from_array(out_tensors(1), out_data, tensor_layout, torch_kCPU)

   ! Load ML model
   call torch_model_load(model, args(1), torch_kCPU)

   ! Infer
   call torch_model_forward(model, in_tensors, out_tensors)

   ! Print the values computed on each MPI rank
   write (6, "('output on rank ',i1,': [',4(f5.1,','),f5.1,']')") rank, out_data(:)

   ! Check output tensor matches expected value
   expected = [(2 * (rank + i), i = 0, 4)]
   test_pass = assert_allclose(out_data, expected, test_name="MPI", rtol=1e-5)

   if (.not. test_pass) then
     call clean_up()
     stop 999
   end if

   if (rank == 0) then
      write (*,*) "MPI Fortran example ran successfully"
   end if

   call clean_up()

  contains

    subroutine clean_up()
      call torch_delete(model)
      call torch_delete(in_tensors)
      call torch_delete(out_tensors)
      call mpi_finalize(ierr)
    end subroutine clean_up

end program inference
