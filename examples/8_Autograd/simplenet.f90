program autograd_simplenet

  ! Import precision info from iso
  use, intrinsic :: iso_fortran_env, only : sp => real32

  ! Import our library for interfacing with PyTorch's Autograd module
  use ftorch, only: torch_kCPU, torch_delete, torch_model, torch_model_load, torch_model_forward,  &
                    torch_tensor, torch_tensor_backward, torch_tensor_from_array, &
                    torch_tensor_get_gradient, assignment(=), operator(*)

  ! Import our tools module for testing utils
  use ftorch_test_utils, only : assert_allclose

  implicit none

  ! Set working precision for reals
  integer, parameter :: wp = sp

  integer, parameter :: ndims = 1
  integer, parameter :: n = 5
  real(wp), dimension(n), target :: in_array, out_array, grad_array
  real(wp), dimension(n) :: expected

  ! Flag for testing
  logical :: test_pass

  ! Set up Torch data structures
  type(torch_model) :: model
  type(torch_tensor) :: in_tensors(1), out_tensors(1), grad_tensors(1), multiplier

  ! Initialise Torch Tensors from input arrays as in Python example
  in_array(:) = [1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp]
  call torch_tensor_from_array(in_tensors(1), in_array, torch_kCPU, requires_grad=.true.)

  ! Initialise Torch Tensors from array used for output
  call torch_tensor_from_array(out_tensors(1), out_array, torch_kCPU) ! , requires_grad=.true.)

  ! ! Load the model and run inference
  call torch_model_load(model, trim("simplenet.pt"), torch_kCPU, requires_grad=.true.)
  call torch_model_forward(model, in_tensors, out_tensors, requires_grad=.true.)

  ! Check the output takes expected values
  expected(:) = [2.0_wp, 4.0_wp, 6.0_wp, 8.0_wp, 10.0_wp]
  if (.not. assert_allclose(out_array, expected, test_name="autograd_simplenet_y")) then
    write(*,*) "Error :: value of y does not match expected value"
    stop 999
  end if

  ! Run the backpropagation operator
  ! This will perform backpropogation on the computational graph from x to y, setting the `grad`
  ! property for x
  call torch_tensor_backward(out_tensors(1))

  ! Create tensors based off output arrays for the gradients and then retrieve them
  call torch_tensor_from_array(grad_tensors(1), grad_array, torch_kCPU)
  call torch_tensor_get_gradient(grad_tensors(1), in_tensors(1))

  ! Check the gradients take expected values
  expected(:) = 2.0
  if (.not. assert_allclose(grad_array, expected, test_name="autograd_simplenet_dydx")) then
    write(*,*) "Error :: value of dydx does not match expected value"
    stop 999
  end if

  call torch_delete(model)
  call torch_delete(in_tensors)
  call torch_delete(out_tensors)
  call torch_delete(grad_tensors)

  write (*,*) "SimpleNet Autograd example ran successfully"

end program autograd_simplenet
