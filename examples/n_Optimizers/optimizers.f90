program example

  ! Import precision info from iso
  use, intrinsic :: iso_fortran_env, only : sp => real32

  ! Import c_int64_t
  use, intrinsic :: iso_c_binding, only: c_int64_t

  ! Import our library for interfacing with PyTorch's Autograd module
  use ftorch, only: assignment(=), operator(-), operator(*), operator(/), operator(**), &
                    torch_kCPU, torch_kFloat32, &
                    torch_tensor, torch_tensor_from_array, &
                    torch_tensor_ones, torch_tensor_empty, &
                    torch_tensor_print, torch_delete, &
                    torch_tensor_backward, torch_tensor_get_gradient, &
                    torch_tensor_mean
  use ftorch_optim, only: torch_optim, torch_optim_SGD, torch_optim_step, torch_optim_zero_grad

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
                        scaling_tensor, scaling_grad, loss, torch_4p0
  type(torch_optim) :: optimizer

  ! Set up training parameters
  integer :: i
  integer, parameter :: n_train = 15
  integer, parameter :: n_print = 1

  ! Initialise Torch Tensors from input/target arrays as in Python example
  input_data = [1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp]
  target_data = [1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp]
  call torch_tensor_from_array(input_vec, input_data, tensor_layout, torch_kCPU)
  call torch_tensor_from_array(target_vec, target_data, tensor_layout, torch_kCPU)

  ! Initialise Scaling tensor as ones as in Python example
  call torch_tensor_ones(scaling_tensor, ndims, tensor_shape, &
                         torch_kFloat32, torch_kCPU, requires_grad=.true.)

  ! Initialise scaling factor of 4.0 for use in tensor operations
  call torch_tensor_from_array(torch_4p0, [4.0_wp], tensor_layout, torch_kCPU)

  ! Initialise an optimizer and apply it to scaling_tensor
  ! TODO optimizer expects an array of tensors, should be a cleaner consistent way to formalise this.
  call torch_optim_SGD(optimizer, [scaling_tensor])

  ! Create an empty loss tensor
  call torch_tensor_empty(scaling_grad, ndims, tensor_shape, torch_kFloat32, torch_kCPU)
  call torch_tensor_empty(loss, ndims, tensor_shape, torch_kFloat32, torch_kCPU)
  ! call torch_tensor_empty(loss, 1, scalar_shape, torch_kFloat32, torch_kCPU)

  ! Conduct training loop
  do i = 1, n_train+1
    write(*,*) "zero_grad"
    ! Zero any previously stored gradients ready for a new iteration
    call torch_optim_zero_grad(optimizer)

    ! Forward pass: multiply the input of ones by the tensor (elementwise)
    write(*,*) "Forward Pass"
    call torch_tensor_from_array(output_vec, output_data, tensor_layout, torch_kCPU)
    output_vec = input_vec * scaling_tensor
    grad_reqd = loss%requires_grad()
    write(*,*) "loss%requires_grad = ", grad_reqd

    ! Populate loss with mean square error (MSE) between target and input
    ! Then perform backward step on loss to propogate gradients using autograd
    write(*,*) "Set Loss"
    loss = ((output_vec - target_vec) ** 2) / torch_4p0
    ! call torch_tensor_mean(loss, (output_vec - target_vec) ** 2)
    grad_reqd = loss%requires_grad()
    write(*,*) "loss%requires_grad = ", grad_reqd

    write(*,*) "Backwards Step"
    call torch_tensor_backward(loss, retain_graph=.true.)
    grad_reqd = loss%requires_grad()
    write(*,*) "loss%requires_grad = ", grad_reqd

    write(*,*) "Get Gradient"
    call torch_tensor_get_gradient(scaling_grad, scaling_tensor)
    grad_reqd = loss%requires_grad()
    write(*,*) "loss%requires_grad = ", grad_reqd
    ! However, we can avoid explicitly passing an initial gradient and instead do this
    ! implicitly by aggregating the loss vector into a scalar value:
    ! TODO: Requires addition of `.mean()` to the FTorch tensor API
    ! loss = ((output - target_vec) ** 2).mean()
    ! loss.backward()

    ! Step the optimizer to update the values in `tensor`
    write(*,*) "Step Optimizer"
    call torch_optim_step(optimizer)
    grad_reqd = loss%requires_grad()
    write(*,*) "loss%requires_grad = ", grad_reqd

    if (modulo(i,n_print) == 0) then
        write(*,*) "================================================"
        write(*,*) "Epoch: ", i
        write(*,*)
        write(*,*) "Output:", output_data
        write(*,*)
        write(*,*) "loss:"
        call torch_tensor_print(loss)
        write(*,*)
        write(*,*) "tensor gradient:"
        call torch_tensor_print(scaling_grad)
        write(*,*)
        write(*,*) "scaling_tensor:"
        call torch_tensor_print(scaling_tensor)
        write(*,*)
    end if

    call torch_delete(output_vec)
    grad_reqd = loss%requires_grad()
    write(*,*) "loss%requires_grad = ", grad_reqd

  end do

  write(*,*) "Training complete."

  ! Clean up created tensors
  call torch_delete(loss)

  write (*,*) "Optimizers example ran successfully"

end program example
