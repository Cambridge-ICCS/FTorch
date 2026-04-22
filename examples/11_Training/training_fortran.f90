program training

  ! Import precision info from iso
  use, intrinsic :: iso_fortran_env, only : sp => real32

  ! Import c_int64_t
  use, intrinsic :: iso_c_binding, only: c_int64_t

  ! Import our library for interfacing with PyTorch's Autograd module
  use ftorch, only: assignment(=), operator(-), operator(*), operator(/), operator(**), &
                    torch_kCPU, torch_kFloat32, &
                    torch_tensor, torch_tensor_from_array, torch_tensor_empty, &
                    torch_tensor_print, &
                    torch_tensor_backward, torch_tensor_get_gradient, &
                    torch_tensor_mean, &
                    torch_model, torch_model_load, torch_model_get_parameters, &
                    torch_model_forward, &
                    torch_optim, torch_optim_SGD

  ! Import our tools module for testing utils
  use ftorch_test_utils, only : allclose

  implicit none

  ! Set working precision for reals
  integer, parameter :: wp = sp

  integer :: num_args, ix
  character(len=128), dimension(:), allocatable :: args

  ! Set up Fortran data structures
  integer, parameter :: n = 5
  real(wp), dimension(n), target :: in_data
  real(wp), dimension(n), target :: out_data
  real(wp), dimension(n), target :: target_data
  real(wp), dimension(1), target :: loss_data

  ! Set up Torch data structures
  type(torch_model) :: model
  type(torch_tensor), dimension(1) :: in_tensors
  type(torch_tensor), dimension(1) :: out_tensors
  type(torch_tensor), dimension(1) :: target_tensors
  integer, parameter :: ndims = 2
  integer(c_int64_t), dimension(ndims), parameter :: weights_shape = [n, n]
  type(torch_tensor), dimension(1) :: weights_tensors
  type(torch_tensor) :: weights_grad
  type(torch_tensor) :: loss
  type(torch_optim) :: optimizer

  ! Set up training parameters
  integer :: i
  integer, parameter :: n_train = 15
  integer, parameter :: n_print = 1

  ! Get TorchScript model file as a command line argument
  num_args = command_argument_count()
  allocate(args(num_args))
  do ix = 1, num_args
    call get_command_argument(ix,args(ix))
  end do

  ! Initialise input data
  in_data = [0.0_wp, 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp]

  ! Specify the output that we want the network to give (a permutation of the input)
  target_data = [4.0_wp, 0.0_wp, 1.0_wp, 2.0_wp, 3.0_wp]

  ! Initialise Torch Tensors from input/output/target arrays
  call torch_tensor_from_array(in_tensors(1), in_data, torch_kCPU)
  call torch_tensor_from_array(out_tensors(1), out_data, torch_kCPU)
  call torch_tensor_from_array(target_tensors(1), target_data, torch_kCPU)

  ! Load ML model with training enabled
  call torch_model_load(model, args(1), torch_kCPU, is_training=.true.)

  ! Get weights from model
  call torch_model_get_parameters(model, weights_tensors)
  write (*,*) "Initial model weights:"
  call torch_tensor_print(weights_tensors(1))

  ! Initialise Scaling tensor as ones as in Python example and a tensor for its gradient
  call torch_tensor_empty(weights_grad, ndims, weights_shape, torch_kFloat32, torch_kCPU)

  ! Initialise an optimizer and apply it to scaling_tensor
  call torch_optim_SGD(optimizer, weights_tensors, learning_rate=1.0D0)

  ! Create a file for recording the loss function progress
  open(unit=10, file="losses_ftorch.dat")

  ! Conduct training loop
  do i = 1, n_train+1
    ! Zero any previously stored gradients ready for a new iteration
    call optimizer%zero_grad()

    ! Forward pass: run inference
    call torch_model_forward(model, in_tensors, out_tensors)

    ! Evaluate the loss function as computed mean square error (MSE) between target and input, then
    ! log its value
    ! NOTE: We need to reconstruct the loss tensor at each iteration to capture a new graph
    !       associated with it as it will be detached after the backward call.
    call torch_tensor_from_array(loss, loss_data, torch_kCPU)
    call torch_tensor_mean(loss, (out_tensors(1) - target_tensors(1)) ** 2)
    write(unit=10, fmt="(es10.4)") loss_data(1)

    ! Perform backward step on loss to propogate gradients using autograd
    ! NOTE: This implicitly passes a unit 'external gradient' to the backward pass
    call torch_tensor_backward(loss)
    call torch_tensor_get_gradient(weights_grad, weights_tensors(1))

    ! Step the optimizer to update the values in `tensor`
    call optimizer%step()

    if (modulo(i,n_print) == 0) then
        write(*,*) "================================================"
        write(*,*) "Epoch: ", i
        write(*,*)
        write(*,*) "Output:", out_data
        write(*,*)
        write(*,*) "loss:"
        call torch_tensor_print(loss)
        write(*,*)
        write(*,*) "tensor gradient:"
        call torch_tensor_print(weights_grad)
        write(*,*)
        write(*,*) "model weights:"
        call torch_tensor_print(weights_tensors(1))
        write(*,*)
    end if

  end do
  close(unit=10)

  ! Check scaling tensor converges to the expected value
  if (.not. allclose(out_data, target_data, test_name="optimizers", rtol=1e-3)) then
    write(*,*) "Error :: value of out_data does not match expected value"
    stop 999
  end if

  write(*,*) "Training complete."

  write (*,*) "Fortran training example ran successfully"

end program training
