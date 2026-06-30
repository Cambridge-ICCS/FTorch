program autograd_simplenet

  ! Import precision info from iso
  use, intrinsic :: iso_fortran_env, only : sp => real32

  ! Import our library for interfacing with PyTorch's Autograd module
  use ftorch, only: torch_kCPU, torch_delete, &
                    torch_tensor, torch_tensor_backward, torch_tensor_from_array, &
                    torch_tensor_get_gradient, torch_tensor_empty, torch_tensor_ones, &
                    torch_tensor_to, torch_tensor_print, &
                    torch_model, torch_model_load, torch_model_forward, torch_model_parameters

  ! Import our tools module for testing utils
  use ftorch_test_utils, only : allclose

  implicit none

  ! Set working precision for reals
  integer, parameter :: wp = sp

  integer, parameter :: ndims = 1
  integer, parameter :: weight_dims = 2
  integer, parameter :: n = 5
  real(wp), dimension(n), target :: in_array, out_array, in_grad_array
  real(wp), dimension(n, n), target :: weights_grad_array
  real(wp), dimension(n) :: expected1d
  real(wp), dimension(n, n) :: expected2d

  character(len=128), parameter :: filename = "simplenet.pt"
  integer :: i

  ! Set up Torch data structures
  type(torch_model) :: model
  type(torch_tensor) :: in_tensors(1), out_tensor, loss_tensors(1), weights_tensors(1)
  type(torch_tensor) :: external_gradient, in_grad_tensor, weights_grad_tensor

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
  expected1d(:) = [2.0_wp, 4.0_wp, 6.0_wp, 8.0_wp, 10.0_wp]
  if (.not. allclose(out_array, expected1d, test_name="autograd_simplenet_y")) then
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

  ! Extract the gradient with respect to the input tensor and check it takes expected values
  call torch_tensor_from_array(in_grad_tensor, in_grad_array, torch_kCPU)
  call torch_tensor_get_gradient(in_grad_tensor, in_tensors(1))
  write(*,*) "dy/dx = ", in_grad_array
  expected1d(:) = 2.0
  if (.not. allclose(in_grad_array, expected1d, test_name="autograd_simplenet_dydx")) then
    write(*,*) "Error :: value of dydx does not match expected value"
    stop 999
  end if

  ! We can also compute gradients with respect to intermediate variables in the Torch model such as
  ! the model weights
  call torch_model_parameters(model, weights_tensors)
  write (*,*) "Model weights:"
  call torch_tensor_print(weights_tensors(1))
  call torch_tensor_from_array(weights_grad_tensor, weights_grad_array, torch_kCPU)
  call torch_tensor_get_gradient(weights_grad_tensor, weights_tensors(1))
  write(*,*) "dy/d(weights):"
  call torch_tensor_print(weights_grad_tensor)

  ! Check the gradient with respect to the weights takes expected values (matrix multiplication)
  do i = 1, 5
    expected2d(i,:) = in_array
  end do
  if (.not. allclose(weights_grad_array, expected2d, test_name="autograd_simplenet_dydw")) then
    write(*,*) "Error :: value of dyd(weights) does not match expected value"
    stop 999
  end if

  ! Cleanup
  call torch_delete(model)
  call torch_delete(in_tensors)
  call torch_delete(loss_tensors)
  call torch_delete(out_tensor)
  call torch_delete(in_grad_tensor)
  call torch_delete(weights_grad_tensor)

  write (*,*) "SimpleNet Autograd example ran successfully"

end program autograd_simplenet
