program example

  ! Import precision info from iso
  use, intrinsic :: iso_fortran_env, only : sp => real32

  ! Import our library for interfacing with PyTorch's Autograd module
  use ftorch

  ! Import our tools module for testing utils
  use utils, only : assert_real_array_1d

  implicit none

  ! Set working precision for reals
  integer, parameter :: wp = sp

  ! Set up Fortran data structures
  real(wp), dimension(2), target :: in_data
  real(wp), dimension(:), pointer :: out_data
  real(wp), dimension(2) :: expected
  integer :: tensor_layout(1) = [1]

  ! Flag for testing
  logical :: test_pass

  ! Set up Torch data structures
  type(torch_tensor) :: a

  ! Construct a Torch Tensor from a Fortran array
  in_data(:) = [2.0, 3.0]
  call torch_tensor_from_array(a, in_data, tensor_layout, torch_kCPU)

  ! Extract a Fortran array from a Torch tensor
  call torch_tensor_to_array(a, out_data, [2])
  write (*,*) "a = ", out_data(:)

  ! Check output tensor matches expected value
  expected = [2.0, 3.0]
  test_pass = assert_real_array_1d(out_data, expected, test_name="torch_tensor_to_array", rtol=1e-5)

  ! Cleanup
  nullify(out_data)
  call torch_tensor_delete(a)

  if (.not. test_pass) then
    stop 999
  end if

end program example
