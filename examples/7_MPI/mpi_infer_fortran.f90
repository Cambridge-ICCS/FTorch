program inference

   ! Import precision info from iso
   use, intrinsic :: iso_fortran_env, only : sp => real32, stdout => output_unit

   ! Import our library for interfacing with PyTorch
   use ftorch, only : torch_model, torch_tensor, torch_kCPU, torch_delete, &
                      torch_tensor_from_array, torch_model_load, torch_model_forward

   ! Import our tools module for testing utils
   use ftorch_test_utils, only : assert_allclose

   ! Import MPI
   use mpi, only : mpi_comm_rank, mpi_comm_size, mpi_comm_world, mpi_finalize, mpi_float, &
                   mpi_gather, mpi_init

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
   integer :: rank, size, ierr, i

   ! Variables for testing
   real(wp), allocatable, dimension(:,:) :: recvbuf
   real(wp), dimension(5) :: result_chk
   integer :: rank_chk

   call mpi_init(ierr)
   call mpi_comm_rank(mpi_comm_world, rank, ierr)
   call mpi_comm_size(mpi_comm_world, size, ierr)

   ! Check MPI was configured correctly
   if (size == 1) then
      write(*,*) "MPI communicator size is 1, indicating that it is not configured correctly"
      write(*,*) "(assuming you specified more than one rank)"
      call clean_up()
      stop 999
   end if

   ! Get TorchScript model file as a command line argument
   num_args = command_argument_count()
   allocate(args(num_args))
   do ix = 1, num_args
      call get_command_argument(ix,args(ix))
   end do

   ! Initialise data and print the values used on each MPI rank
   in_data = [(rank + i, i = 0, 4)]
   write(unit=stdout, fmt="('input on rank ',i1,': ')", advance="no") rank
   write(unit=stdout, fmt=100) in_data(:)
   100 format('[',4(f5.1,','),f5.1,']')

   ! Create Torch input/output tensors from the above arrays
   call torch_tensor_from_array(in_tensors(1), in_data, tensor_layout, torch_kCPU)
   call torch_tensor_from_array(out_tensors(1), out_data, tensor_layout, torch_kCPU)

   ! Load ML model
   call torch_model_load(model, args(1), torch_kCPU)

   ! Run inference on each MPI rank
   call torch_model_forward(model, in_tensors, out_tensors)

   ! Print the values computed on each MPI rank
   write(unit=stdout, fmt="('output on rank ',i1,': ')", advance="no") rank
   write(unit=stdout, fmt=100) out_data(:)

   ! Gather the outputs onto rank 0
   allocate(recvbuf(5,size))
   call mpi_gather(out_data, 5, mpi_float, recvbuf, 5, mpi_float, 0, mpi_comm_world, ierr)

   ! Check that the correct values were attained
   if (rank == 0) then

      ! Check output tensor matches expected value
      do rank_chk = 0, size-1
        expected = [(2 * (rank_chk + i), i = 0, 4)]
        result_chk(:) = recvbuf(:,rank_chk+1)
        test_pass = assert_allclose(result_chk, expected, test_name="MPI")
        if (.not. test_pass) then
          write(unit=stdout, fmt="('rank ',i1,' result: ')") rank_chk
          write(unit=stdout, fmt=100) result_chk(:)
          write(unit=stdout, fmt="('does not match expected value')")
          write(unit=stdout, fmt=100) expected(:)
          call clean_up()
          stop 999
        end if
      end do

      write (*,*) "MPI Fortran example ran successfully"
   end if

   call clean_up()

  contains

    subroutine clean_up()
      call torch_delete(model)
      call torch_delete(in_tensors)
      call torch_delete(out_tensors)
      call mpi_finalize(ierr)
      deallocate(recvbuf)
    end subroutine clean_up

end program inference
