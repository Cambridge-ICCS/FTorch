program example

  ! Import precision info from iso
  use, intrinsic :: iso_fortran_env, only : sp => real32

  ! Import c_int64_t
  use, intrinsic :: iso_c_binding, only: c_int64_t

  ! Import our library for interfacing with PyTorch's Autograd module
  use ftorch, only: assignment(=), operator(-), operator(*), operator(**), &
                    torch_kCPU, torch_kFloat32, &
                    torch_tensor, torch_tensor_from_array, &
                    torch_tensor_ones, torch_tensor_empty, &
                    torch_delete, torch_tensor_backward

  implicit none

  ! Set working precision for reals
  integer, parameter :: wp = sp

  ! Set up Fortran data structures
  integer, parameter :: ndims = 1
  integer, parameter :: n=4
  real(wp), dimension(n), target :: input_data, output_data, target_data
  integer :: tensor_layout(ndims) = [1]
  logical :: grad_reqd

  ! Set up Torch data structures
  integer(c_int64_t), dimension(1), parameter :: tensor_shape = [4]
  integer(c_int64_t), dimension(1), parameter :: scalar_shape = [1]
  type(torch_tensor) :: input_vec, output_vec, target_vec, &
                        scaling_tensor, loss

  ! Set up training parameters
  integer :: i
  integer, parameter :: n_train = 15
  integer, parameter :: n_print = 1

  ! Initialise Torch Tensors from input/target arrays as in Python example
  input_data = [1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp]
  target_data = [1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp]
  call torch_tensor_from_array(input_vec, input_data, tensor_layout, torch_kCPU)
  call torch_tensor_from_array(output_vec, output_data, tensor_layout, torch_kCPU)
  call torch_tensor_from_array(target_vec, target_data, tensor_layout, torch_kCPU)

  ! Initialise Scaling tensor as ones as in Python example
  call torch_tensor_ones(scaling_tensor, ndims, tensor_shape, &
                         torch_kFloat32, torch_kCPU, requires_grad=.true.)

  ! Create an empty loss tensor
  call torch_tensor_empty(loss, ndims, tensor_shape, torch_kFloat32, torch_kCPU)


  write(*,*) "Start of code"
  grad_reqd = loss%requires_grad()
  write(*,*) "loss%requires_grad = ", grad_reqd

  ! Conduct training loop
  do i = 1, 3

    write(*,*) " "
    write(*,*) "============================================ "
    write(*,*) "ITERATION: ", i
    write(*,*) "============================================ "
    write(*,*) " "

    write(*,*) "calculate output"
    output_vec = input_vec * scaling_tensor
    grad_reqd = loss%requires_grad()
    write(*,*) "loss%requires_grad = ", grad_reqd

    ! Populate loss with mean square error (MSE) between target and input
    ! Then perform backward step on loss to propogate gradients using autograd
    write(*,*) "Set Loss"
    loss = ((output_vec - target_vec) ** 2)
    ! call torch_tensor_mean(loss, (output_vec - target_vec) ** 2)
    grad_reqd = loss%requires_grad()
    write(*,*) "loss%requires_grad = ", grad_reqd

    write(*,*) "Backwards Step"
    call torch_tensor_backward(loss, retain_graph=.true.)
    grad_reqd = loss%requires_grad()
    write(*,*) "loss%requires_grad = ", grad_reqd

  end do

  ! Clean up created tensors
  call torch_delete(output_vec)
  call torch_delete(loss)

  write (*,*) "rgmwe example ran successfully"

end program example
