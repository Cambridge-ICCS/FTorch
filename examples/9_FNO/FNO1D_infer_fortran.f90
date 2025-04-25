program inference

   ! Import precision info from iso
   use, intrinsic :: iso_fortran_env, only : sp => real32

   ! Import our library for interfacing with PyTorch
   use ftorch, only : torch_model, torch_tensor, torch_kCPU, torch_delete, &
                      torch_tensor_from_array, torch_model_load, torch_model_forward

   ! Import our tools module for testing utils
   use ftorch_test_utils, only : assert_allclose

   implicit none

   integer, parameter :: wp = sp

   integer, parameter :: in_dims = 3
   integer, parameter :: in_shape(in_dims) = [1, 32, 2]
   integer, parameter :: out_dims = 3
   integer, parameter :: out_shape(out_dims) = [1, 32, 1]
   real(wp) :: x_real, true_sine, prediction, error
   real(wp), parameter :: tol = 0.05  ! <-- Set your acceptable error tolerance

   integer :: num_args, ix, i
   character(len=128), dimension(:), allocatable :: args

   ! Set up Fortran data structures
   real(wp), allocatable, dimension(:,:,:) :: in_data
   real(wp), allocatable, dimension(:,:,:) :: out_data

   ! Set up Torch data structures
   ! The net, a vector of input tensors (in this case we only have one), and the output tensor
   type(torch_model) :: model
   type(torch_tensor), dimension(1) :: in_tensors
   type(torch_tensor), dimension(1) :: out_tensors

   ! Flag for testing
   logical :: test_pass

   ! Get TorchScript model file as a command line argument
   num_args = command_argument_count()
   allocate(args(num_args))
   do ix = 1, num_args
       call get_command_argument(ix,args(ix))
   end do

   allocate(in_data(in_shape(1), in_shape(2), in_shape(3)))
   allocate(out_data(out_shape(1), out_shape(2), out_shape(3)))

   do i = 1, 32
     in_data(1, i, 1) = 0.0                          ! dummy input
     in_data(1, i, 2) = real(i - 1) / real(31)       ! grid input: linspace
   end do

   ! Initialise data
   ! in_data = [0.0_wp, 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp]

   ! Create Torch input/output tensors from the above arrays
   call torch_tensor_from_array(in_tensors(1), in_data, torch_kCPU)
   call torch_tensor_from_array(out_tensors(1), out_data, torch_kCPU)



   ! Load ML model
   call torch_model_load(model, args(1), torch_kCPU)

   ! Check shape of in_tensors
   print *, "Input array shape: ", shape(in_data)
   ! call torch_tensor_shape(in_tensors(1), dims, shape)
   ! print *, "Input tensor shape: ", shape(1:dims)


   ! Infer
   call torch_model_forward(model, in_tensors, out_tensors)

   ! write (*,*) out_data(:)

   ! Check output tensor matches expected value
   test_pass = .true.
   do i = 1, 32
     x_real = real(i-1) / real(31)
     true_sine = sin(2.0 * 3.14159265 * x_real)
     prediction = out_data(1, i, 1)

     error = abs(prediction - true_sine)

     if (error > tol) then
       print *, "FAILED at x=", x_real, &
         ": prediction=", prediction, &
         " true=", true_sine, &
         " error=", error
       test_pass = .false.
     end if
   end do

   if (test_pass) then
     print *, "All predictions within tolerance."
   else
     print *, "Some predictions exceeded tolerance."
   end if


   ! expected = [0.0_wp, 2.0_wp, 4.0_wp, 6.0_wp, 8.0_wp]
   ! test_pass = assert_allclose(out_data, expected, test_name="FNO1d", rtol=1e-5)

   ! Cleanup
   call torch_delete(model)
   call torch_delete(in_tensors)
   call torch_delete(out_tensors)

   if (.not. test_pass) then
     stop 999
   end if

  write (*,*) "FNO1d example ran successfully"

end program inference
