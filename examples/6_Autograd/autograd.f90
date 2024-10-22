program example

  ! Import precision info from iso
  use, intrinsic :: iso_fortran_env, only : sp => real32

  ! Import our library for interfacing with PyTorch's Autograd module
  use ftorch

  implicit none

  ! Set working precision for reals
  integer, parameter :: wp = sp

  ! Set up Fortran data structures
  integer, parameter :: n=2, m=5
  real(wp), dimension(n,m), target :: in_data
  real(wp), dimension(:,:), pointer :: out_data
  integer :: tensor_layout(2) = [1, 2]
  integer :: i, j

  ! Set up Torch data structures
  type(torch_tensor) :: tensor

  ! initialize in_data with some fake data
  do j = 1, m
    do i = 1, n
      in_data(i,j) = ((i-1)*m + j) * 1.0_wp
    end do
  end do

  ! Construct a Torch Tensor from a Fortran array
  call torch_tensor_from_array(tensor, in_data, tensor_layout, torch_kCPU)

  ! check tensor rank and shape match those of in_data
  if (tensor%get_rank() /= 2) then
    print *, "Error :: rank should be 2"
    stop 1
  end if
  if (any(tensor%get_shape() /= [2, 5])) then
    print *, "Error :: shape should be (2, 5)"
    stop 1
  end if

  ! Extract a Fortran array from a Torch tensor
  call torch_tensor_to_array(tensor, out_data, shape(in_data))

  ! check that the data match
  if (any(in_data /= out_data)) then
    print *, "Error :: in_data does not match out_data"
    stop 1
  end if

  ! Cleanup
  nullify(out_data)
  call torch_tensor_delete(tensor)

  write (*,*) "test completed successfully"

end program example
