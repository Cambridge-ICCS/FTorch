program inference

   ! Import precision info from iso
   use, intrinsic :: iso_fortran_env, only : sp => real32

   ! Import our library for interfacing with PyTorch
   use ftorch, only : torch_model, torch_tensor, &
                      torch_kCPU, torch_kCUDA, torch_kXPU, torch_kMPS, &
                      torch_tensor_from_array, torch_model_load, torch_model_forward, &
                      torch_delete

   ! Import our tools module for testing utils
   use ftorch_test_utils, only : assert_allclose

   implicit none

   ! Set precision for reals
   integer, parameter :: wp = sp

   integer :: num_args, ix
   character(len=128), dimension(:), allocatable :: args

   ! Set up Fortran data structures
   real(wp), dimension(5), target :: in_data
   real(wp), dimension(5), target :: out_data
   real(wp), dimension(5) :: expected
   integer, parameter :: tensor_layout(1) = [1]

   ! Set up Torch data structures
   type(torch_model) :: model
   type(torch_tensor), dimension(1) :: in_tensors
   type(torch_tensor), dimension(1) :: out_tensors

   ! Variables for multi-GPU setup
   integer :: num_devices = 2
   integer :: device_type, device_index, i

   ! Flag for testing
   logical :: test_pass

   ! Get device type as first command line argument and TorchScript model file as second command
   ! line argument
   num_args = command_argument_count()
   allocate(args(num_args))
   do ix = 1, num_args
      call get_command_argument(ix,args(ix))
   end do
   if (trim(args(1)) == "cuda") then
      device_type = torch_kCUDA
   else if (trim(args(1)) == "xpu") then
      device_type = torch_kXPU
   else if (trim(args(1)) == "mps") then
      device_type = torch_kMPS
      num_devices = 1
   else
      write (*,*) "Error :: invalid device type", trim(args(1))
      stop 999
   end if

   do device_index = 0, num_devices-1

      ! Initialise data and print the values used
      in_data = [(device_index + i, i = 0, 4)]
      write (6, 100) device_index, in_data(:)
      100 format("input on device ", i1,": [", 4(f5.1,","), f5.1,"]")

      ! Create Torch input tensor from the above array and assign it to the first (and only)
      ! element in the array of input tensors.
      ! We use the specified GPU device type with the given device index
      call torch_tensor_from_array(in_tensors(1), in_data, tensor_layout, device_type, &
                                   device_index=device_index)

      ! Create Torch output tensor from the above array.
      ! Here we use the torch_kCPU device type since the tensor is for output only
      ! i.e. to be subsequently used by Fortran on CPU.
      call torch_tensor_from_array(out_tensors(1), out_data, tensor_layout, torch_kCPU)

      ! Load ML model. Ensure that the same device type and device index are used
      ! as for the input data.
      call torch_model_load(model, args(2), device_type, device_index=device_index)

      ! Infer
      call torch_model_forward(model, in_tensors, out_tensors)

      ! Print the values computed on the current device.
      write (6, 200) device_index, out_data(:)
      200 format("output on device ", i1,": [", 4(f5.1,","), f5.1,"]")

      ! Check output tensor matches expected value
      expected = [(2 * (device_index + i), i = 0, 4)]
      test_pass = assert_allclose(out_data, expected, test_name="MultiGPU")

      ! Cleanup
      call torch_delete(model)
      call torch_delete(in_tensors)
      call torch_delete(out_tensors)

      if (.not. test_pass) then
        stop 999
      end if

   end do

   write (*,*) "MultiGPU example ran successfully"

end program inference
