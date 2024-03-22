program test_device_index

! Import precision info from iso
use, intrinsic :: iso_fortran_env, only : sp => real32

! Import our library for interfacing with PyTorch
use ftorch

! Import MPI
use mpi

implicit none

! Set precision for reals
integer, parameter :: wp = sp

! Set up Fortran data structures
real(wp), dimension(5), target :: in_data
integer :: tensor_layout(1) = [1]

! Set up Torch data structures
type(torch_tensor) :: in_tensor
integer :: device_type
integer :: device_index

! MPI configuration
integer rank, ierr

call mpi_init(ierr)
call mpi_comm_rank(mpi_comm_world, rank, ierr)

! Initialise data
in_data = [0.0, 1.0, 2.0, 3.0, 4.0]

! Loop over device type torch_kCPU and torch_kGPU
do device_type = 0, 1
  if (device_type == torch_kCPU) then
    device_index = - 1
  else
    device_index = rank
  end if

  ! Create Torch input tensor from the above arrays
  in_tensor = torch_tensor_from_array(in_data, tensor_layout, device_type, device_index)

  ! Print some information
  if (torch_tensor_get_device_index(in_tensor) == device_index) then
    write(*, *) rank, "PASS"
  else
    write(*, *) rank, "expected index ", device_index, "got ", torch_tensor_get_device_index(in_tensor)
  end if

  ! Cleanup
  call torch_tensor_delete(in_tensor)
end do
call mpi_finalize(ierr)

end program test_device_index
