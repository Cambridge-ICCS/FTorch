program foptimizer

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
  use ftorch_optim, only: torch_optim, torch_optim_SGD

  ! Import our tools module for testing utils
  use ftorch_test_utils, only : assert_allclose

  implicit none

  ! Set working precision for reals
  integer, parameter :: wp = sp

  ! Set up Fortran data structures
  integer, parameter :: ndims = 1
  integer, parameter :: n=4
  real(wp), dimension(n), target :: input_data, output_data, target_data, scaling_data
  real(wp), dimension(1), target :: loss_data
  integer :: scalar_layout(1) = [1]
  integer :: tensor_layout(ndims) = [1]

  ! Set up Torch data structures
  integer(c_int64_t), dimension(1), parameter :: tensor_shape = [4]
  type(torch_tensor) :: input_vec, output_vec, target_vec, scaling_tensor, scaling_grad, loss
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

  ! Initialise Scaling tensor as ones as in Python example and a tensor for its gradient
  scaling_data(:) = 1.0_wp
  call torch_tensor_from_array(scaling_tensor, scaling_data, tensor_layout, torch_kCPU, &
                               requires_grad=.true.)
  call torch_tensor_empty(scaling_grad, ndims, tensor_shape, torch_kFloat32, torch_kCPU)

  ! Initialise an optimizer and apply it to scaling_tensor
  ! NOTE: The optimizer expects an array of tensors.
  call torch_optim_SGD(optimizer, [scaling_tensor], learning_rate=1D0)

  ! Create a file for recording the loss function progress
  open(unit=10, file="losses_ftorch.dat")

  ! Conduct training loop
  do i = 1, n_train+1
    ! Zero any previously stored gradients ready for a new iteration
    call optimizer%zero_grad()

    ! Forward pass: multiply the input of ones by the tensor (elementwise)
    ! Create a tensor to extract the output of the operation we wish to optimize
    ! NOTE: We need to reconstruct the output tensor at each iteration to capture a new graph
    !       associated with it, as it will be detached after a backward call.
    call torch_tensor_from_array(output_vec, output_data, tensor_layout, torch_kCPU)
    output_vec = input_vec * scaling_tensor

    ! Evaluate the loss function as computed mean square error (MSE) between target and input, then
    ! log its value
    ! NOTE: We need to reconstruct the loss tensor at each iteration to capture a new graph
    !       associated with it, as it will be detached after a backward call.
    call torch_tensor_from_array(loss, loss_data, scalar_layout, torch_kCPU)
    call torch_tensor_mean(loss, (output_vec - target_vec) ** 2)
    write(unit=10, fmt="(e9.4)") loss_data(1)

    ! Perform backward step on loss to propogate gradients using autograd
    ! NOTE: This implicitly passes a unit 'external gradient' to the backward pass
    call torch_tensor_backward(loss)
    call torch_tensor_get_gradient(scaling_grad, scaling_tensor)

    ! Step the optimizer to update the values in `tensor`
    call optimizer%step()

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

    ! Clean up created tensors
    call torch_delete(output_vec)
    call torch_delete(loss)

  end do
  close(unit=10)

  ! Check scaling tensor converges to the expected value
  if (.not. assert_allclose(scaling_data, target_data, test_name="optimizers", rtol=1e-3)) then
    write(*,*) "Error :: value of scaling_data does not match expected value"
    stop 999
  end if

  write(*,*) "Training complete."

  write (*,*) "Fortran optimizers example ran successfully"

end program foptimizer
