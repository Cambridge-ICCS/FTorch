program example

  ! Import precision info from iso
  use, intrinsic :: iso_fortran_env, only : sp => real32

  ! Import our library for interfacing with PyTorch's Autograd module
  use ftorch, only: assignment(=), operator(+), operator(-), operator(*), &
    operator(/), operator(**), torch_kCPU, torch_kFloat32, torch_tensor, torch_tensor_delete, &
    torch_tensor_empty, torch_tensor_from_array

  ! Import our tools module for testing utils
  use ftorch_test_utils, only : assert_allclose

  implicit none

  ! Set working precision for reals
  integer, parameter :: wp = sp

  ! Set up Fortran data structures
  integer, parameter :: ndims = 2
  integer, parameter :: n=2, m=1
  real(wp), dimension(n,m), target :: in_data1
  real(wp), dimension(n,m), target :: in_data2
  real(wp), dimension(n,m), target :: out_data
  real(wp), dimension(n,m) :: expected
  integer :: tensor_layout(ndims) = [1, 2]

  ! Flag for testing
  logical :: test_pass

  ! Set up Torch data structures
  type(torch_tensor) :: a, b, Q

  ! Initialise Torch Tensors from input arrays as in Python example
  in_data1(:,1) = [2.0_wp, 3.0_wp]
  in_data2(:,1) = [6.0_wp, 4.0_wp]
  ! TODO: Implement requires_grad=.true.
  call torch_tensor_from_array(a, in_data1, tensor_layout, torch_kCPU)
  call torch_tensor_from_array(b, in_data2, tensor_layout, torch_kCPU)

  ! Initialise Torch Tensor from array used for output
  call torch_tensor_from_array(Q, out_data, tensor_layout, torch_kCPU)

  ! Check arithmetic operations work for torch_tensors
  write (*,*) "a = ", in_data1(:,1)
  write (*,*) "b = ", in_data2(:,1)
  Q = 3 * (a**3 - b * b / 3)

  ! Extract a Fortran array from a Torch tensor
  write (*,*) "Q = 3 * (a ** 3 - b * b / 2) =", out_data(:,1)

  ! Check output tensor matches expected value
  expected(:,1) = [-12.0_wp, 65.0_wp]
  test_pass = assert_allclose(out_data, expected, test_name="autograd_Q")
  if (.not. test_pass) then
    print *, "Error :: out_data does not match expected value"
    stop 999
  end if

  ! Back-propagation
  ! TODO: Requires API extension

  write (*,*) "Autograd example ran successfully"

end program example
