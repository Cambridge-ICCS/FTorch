program autograd_simplenet

  ! Import precision info from iso
  use, intrinsic :: iso_fortran_env, only : sp => real32

  ! Import our library for interfacing with PyTorch's Autograd module
  use ftorch, only: torch_kCPU, torch_delete, torch_model, torch_model_load, torch_model_forward,  &
                    torch_tensor, torch_tensor_backward, torch_tensor_from_array, &
                    torch_tensor_get_gradient, torch_tensor_empty, torch_tensor_ones, &
                    torch_tensor_to

  ! Import our tools module for testing utils
  use ftorch_test_utils, only : allclose

  implicit none

  ! Set working precision for reals
  integer, parameter :: wp = sp

  integer, parameter :: ndims = 1
  integer, parameter :: n = 5
  real(wp), dimension(n), target :: in_array, out_array, grad_array
  real(wp), dimension(n) :: expected

  character(len=128), parameter :: filename = "simplenet.pt"

  ! Set up Torch data structures
  type(torch_model) :: model
  type(torch_tensor) :: in_tensors(1), out_tensor, loss_tensors(1), grad_tensor
  type(torch_tensor) :: multiplier, external_gradient

  ! Initialise Torch Tensors from input arrays as in Python example
  in_array(:) = [1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp]
  write(*,*) "x = ", in_array
  call torch_tensor_from_array(in_tensors(1), in_array, torch_kCPU, requires_grad=.true.)

  ! Initialise Torch Tensors for the model outputs, typically referred to as the model loss
  call torch_tensor_empty(loss_tensors(1), ndims, in_tensors(1)%get_shape(), &
                          in_tensors(1)%get_dtype(), torch_kCPU)

  ! Load the model from file
  call torch_model_load(model, trim(filename), torch_kCPU)

  ! Propagate the input tensor through the model with requires_grad=.true.
  ! NOTE: Calling torch_model_forward with requires_grad=.true. produces output tensors that point
  !       to the model output in Torch. That is, any association with Fortran data is lost. This is
  !       required for backpropogation to function correctly when we differentiate with respect to
  !       model parameters
  call torch_model_forward(model, in_tensors, loss_tensors, requires_grad=.true.)

  ! Initialise Torch Tensors from array used for output and copy the loss values over to check their
  ! values
  call torch_tensor_from_array(out_tensor, out_array, torch_kCPU)
  call torch_tensor_to(loss_tensors(1), out_tensor)
  write(*,*) "y = ", out_array
  expected(:) = [2.0_wp, 4.0_wp, 6.0_wp, 8.0_wp, 10.0_wp]
  if (.not. allclose(out_array, expected, test_name="autograd_simplenet_y")) then
    write(*,*) "Error :: value of y does not match expected value"
    stop 999
  end if

  ! Create an appropriate external gradient tensor
  ! You can think of this as the direction in which the derivative is computed. Here we exploit the
  ! fact that we know that the mixed partial derivatives of the output are zero (i.e., each
  ! component of the output depends only on the corresponding component of the input) so can compute
  ! the full gradient in one backpropagation by choosing a tensor filled with ones.
  ! NOTE: An external gradient is required when calling backpropagation on a torch_tensor, except
  !       when it is scalar-valued. In that case, the external gradient defaults to one.
  call torch_tensor_ones(external_gradient, ndims, loss_tensors(1)%get_shape(), &
                         loss_tensors(1)%get_dtype(), torch_kCPU)

  ! Run the backpropagation operator
  ! This will perform backpropogation on the computational graph from x to y, setting the `grad`
  ! property for x
  call torch_tensor_backward(loss_tensors(1), external_gradient)

  ! Create an array based of the gradient tensor to hold gradient information
  call torch_tensor_from_array(grad_tensor, grad_array, torch_kCPU)

  ! Specify that we want the gradient with respect to the input tensor
  call torch_tensor_get_gradient(grad_tensor, in_tensors(1))

  ! Check the gradient array takes expected values
  write(*,*) "dy/dx = ", grad_array
  expected(:) = 2.0
  if (.not. allclose(grad_array, expected, test_name="autograd_simplenet_dydx")) then
    write(*,*) "Error :: value of dydx does not match expected value"
    stop 999
  end if

  call torch_delete(model)
  call torch_delete(in_tensors)
  call torch_delete(loss_tensors)
  call torch_delete(out_tensor)
  call torch_delete(grad_tensor)

  write (*,*) "SimpleNet Autograd example ran successfully"

end program autograd_simplenet
