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
  real(wp), dimension(n), target :: out_data1, out_data2, out_data3
  real(wp), dimension(n) :: expected
  integer :: tensor_layout(ndims) = [1]

  ! Flag for testing
  logical :: test_pass

  ! Set up Torch data structures
  type(torch_tensor) :: a, b, Q, multiplier, divisor, dQda, dQdb

  ! Initialise Torch Tensors from input arrays as in Python example
  call torch_tensor_from_array(a, [2.0_wp, 3.0_wp], tensor_layout, torch_kCPU, requires_grad=.true.)
  call torch_tensor_from_array(b, [6.0_wp, 4.0_wp], tensor_layout, torch_kCPU, requires_grad=.true.)

  ! Initialise Torch Tensor from array used for output
  call torch_tensor_from_array(Q, out_data1, tensor_layout, torch_kCPU)

  ! Scalar multiplication and division are not currently implemented in FTorch. However, you can
  ! achieve the same thing by defining a rank-1 tensor with a single entry, as follows:
  call torch_tensor_from_array(multiplier, [3.0_wp], tensor_layout, torch_kCPU)
  call torch_tensor_from_array(divisor, [3.0_wp], tensor_layout, torch_kCPU)

  ! Compute the same mathematical expression as in the Python example
  Q = multiplier * (a**3 - b * b / divisor)
  write (*,*) "Q = 3 * (a^3 - b*b/3) = 3*a^3 - b^2 = ", out_data1(:)

  ! Check output tensor matches expected value
  expected(:) = [-12.0_wp, 65.0_wp]
  if (.not. assert_allclose(out_data1, expected, test_name="autograd_Q")) then
    write(*,*) "Error :: value of Q does not match expected value"
    stop 999
  end if

  ! Run the back-propagation operator
  ! This will perform backpropogation on the tensors involved in generating Q (a and b), setting the `grad` property for both of them.
  call torch_tensor_backward(Q)

  ! Create tensors based off output arrays for the gradients and then retrieve them
  call torch_tensor_from_array(dQda, out_data2, tensor_layout, torch_kCPU)
  call torch_tensor_from_array(dQdb, out_data3, tensor_layout, torch_kCPU)
  dQda = a%grad()
  dQdb = b%grad()

  ! Check the gradients take expected values
  write(*,*) "dQda = 9*a^2 = ", out_data2
  expected(:) = [36.0_wp, 81.0_wp]
  if (.not. assert_allclose(out_data2, expected, test_name="autograd_dQdb")) then
    write(*,*) "Error :: value of dQdb does not match expected value"
    stop 999
  end if
  write(*,*) "dQdb = - 2*b = ", out_data3
  expected(:) = [-12.0_wp, -8.0_wp]
  if (.not. assert_allclose(out_data3, expected, test_name="autograd_dQdb")) then
    write(*,*) "Error :: value of dQdb does not match expected value"
    stop 999
  end if

  write (*,*) "Autograd example ran successfully"

end program example
