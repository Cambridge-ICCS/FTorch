program example

  ! Import precision info from iso
  use, intrinsic :: iso_fortran_env, only : sp => real32

  ! Import our library for interfacing with PyTorch's Autograd module
  use ftorch, only: assignment(=), operator(+), operator(-), operator(*), operator(/), &
                    operator(**), torch_kCPU, torch_tensor, torch_tensor_backward, &
                    torch_tensor_from_array

  ! Import our tools module for testing utils
  use ftorch_test_utils, only : assert_allclose

  implicit none

  ! Set working precision for reals
  integer, parameter :: wp = sp

  integer, parameter :: ndims = 1
  integer, parameter :: n = 2
  real(wp), dimension(n), target :: out_data
  real(wp), dimension(n) :: expected
  integer :: tensor_layout(ndims) = [1]

  ! Flag for testing
  logical :: test_pass

  ! Set up Torch data structures
  type(torch_tensor) :: a, b, Q, dQda, dQdb, multiplier, divisor

  ! Initialise Torch Tensors from input arrays as in Python example
  call torch_tensor_from_array(a, [2.0_wp, 3.0_wp], tensor_layout, torch_kCPU, requires_grad=.true.)
  call torch_tensor_from_array(b, [6.0_wp, 4.0_wp], tensor_layout, torch_kCPU, requires_grad=.true.)

  ! Initialise Torch Tensor from array used for output
  call torch_tensor_from_array(Q, out_data, tensor_layout, torch_kCPU)

  ! Scalar multiplication and division are not currently implemented in FTorch. However, you can
  ! achieve the same thing by defining a rank-1 tensor with a single entry, as follows:
  call torch_tensor_from_array(multiplier, [3.0_wp], tensor_layout, torch_kCPU)
  call torch_tensor_from_array(divisor, [3.0_wp], tensor_layout, torch_kCPU)

  ! Compute the same mathematical expression as in the Python example
  Q = multiplier * (a**3 - b * b / divisor)
  write (*,*) "Q = 3 * (a^3 - b*b/3) = 3*a^3 - b^2 = ", out_data(:)

  ! Check output tensor matches expected value
  if (.not. assert_allclose(out_data, [-12.0_wp, 65.0_wp], test_name="autograd_Q")) then
    print *, "Error :: value of Q does not match expected value"
    stop 999
  end if

  ! Run the back-propagation operator
  call torch_tensor_backward(Q)
  dQda = a%grad()
  dQdb = b%grad()

  ! Extract Fortran arrays from the Torch tensors and check the gradients take expected values
  call torch_tensor_to_array(dQda, out_data2, [2])
  print *, "dQda = 9*a^2 = ", out_data2
  if (.not. assert_allclose(out_data2, [36.0_wp, 81.0_wp], test_name="autograd_dQdb")) then
    call clean_up()
    print *, "Error :: value of dQdb does not match expected value"
    stop 999
  end if
  call torch_tensor_to_array(dQdb, out_data3, [2])
  print *, "dQdb = - 2*b = ", out_data3
  if (.not. assert_allclose(out_data3, [-12.0_wp, -8.0_wp], test_name="autograd_dQdb")) then
    call clean_up()
    print *, "Error :: value of dQdb does not match expected value"
    stop 999
  end if

  write (*,*) "Autograd example ran successfully"

end program example
