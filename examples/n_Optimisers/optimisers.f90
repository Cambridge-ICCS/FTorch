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
                    torch_tensor_print, torch_delete
  use ftorch_optim, only: torch_optim, torch_optim_SGD, torch_optim_step

  implicit none

  ! Set working precision for reals
  integer, parameter :: wp = sp

  ! Set up Fortran data structures
  integer, parameter :: ndims = 1
  integer, parameter :: n=4
  real(wp), dimension(n), target :: input_data, output_data, target_data
  integer :: tensor_layout(ndims) = [1]

  ! Set up Torch data structures
  integer(c_int64_t), dimension(1), parameter :: tensor_shape = [4]
  type(torch_tensor) :: input_vec, output_vec, target_vec, scaling_tensor, loss, torch_4p0
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
  call torch_tensor_from_array(torch_4p0, [4.0_wp], tensor_layout, torch_kCPU, requires_grad=.true.)

  ! Initialise an optimiser and apply it to scaling_tensor
  ! TODO optimizer expects an array of tensors, should be a cleaner consistent way to formalise this.
  call torch_optim_SGD(optimizer, [scaling_tensor], learning_rate=1D0)

  ! Conduct training loop
  do i = 1, n_train+1
    ! Zero any previously stored gradients ready for a new iteration
    ! TODO: implement equivalent to optimizer.zero_grad()

    ! Forward pass: multiply the input of ones by the tensor (elementwise)
    call torch_tensor_from_array(output_vec, output_data, tensor_layout, torch_kCPU)
    output_vec = input_vec * scaling_tensor

    ! Create an empty loss tensor and populate with mean square error (MSE) between target and input
    ! Then perform backward step on loss to propogate gradients using autograd
    !
    ! We could use the following lines to do this by explicitly specifying a
    ! gradient of ones to start the process:
    call torch_tensor_empty(loss, ndims, tensor_shape, &
                           torch_kFloat32, torch_kCPU)
    loss = ((output_vec - target_vec) ** 2) / torch_4p0
    ! TODO: add in backpropogation functionality for loss.backward(gradient=torch.ones(4))
    !
    ! However, we can avoid explicitly passing an initial gradient and instead do this
    ! implicitly by aggregating the loss vector into a scalar value:
    ! TODO: Requires addition of `.mean()` to the FTorch tensor API
    ! loss = ((output - target_vec) ** 2).mean()
    ! loss.backward()

    ! Step the optimiser to update the values in `tensor`
    call torch_optim_step(optimizer)

    if (modulo(i,n_print) == 0) then
        write(*,*) "================================================"
        write(*,*) "Epoch: ", i
        write(*,*)
        write(*,*) "Output:", output_data
        write(*,*)
        write(*,*) "loss:"
        call torch_tensor_print(loss)
        write(*,*)
        write(*,*) "tensor gradient: TODO: scaling_tensor.grad"
        write(*,*)
        write(*,*) "scaling_tensor:"
        call torch_tensor_print(scaling_tensor)
        write(*,*)
    end if

    ! Clean up created tensors
    call torch_delete(output_vec)
    call torch_delete(loss)

  end do

  write(*,*) "Training complete."

  write (*,*) "Optimisers example ran successfully"

end program example
