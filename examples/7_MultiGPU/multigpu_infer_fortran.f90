program inference

   ! Import precision info from iso
   use, intrinsic :: iso_fortran_env, only : sp => real32, stdout => output_unit

   ! Import our library for interfacing with PyTorch
   use ftorch, only : torch_model, torch_tensor, torch_kCPU, &
                      torch_tensor_from_array, torch_model_load, torch_model_forward, &
                      torch_delete

   ! Import our tools module for testing utils
   use ftorch_test_utils, only : allclose

   implicit none

   ! Set precision for reals
   integer, parameter :: wp = sp

   integer :: num_args, ix
   character(len=128), dimension(:), allocatable :: args
   character(len=128) :: device_code
   character(len=128) :: num_devices_str
   integer :: torch_device
   integer :: num_devices

   ! Set up Fortran data structures
   real(wp), dimension(5), target :: in_data
   real(wp), dimension(5), target :: out_data
   real(wp), dimension(5) :: expected

   ! Set up Torch data structures
   type(torch_model) :: model
   type(torch_tensor), dimension(1) :: in_tensors
   type(torch_tensor), dimension(1) :: out_tensors

   integer :: device_index
   integer :: i

   ! Flag for testing
   logical :: test_pass

   ! Get device type as first command line argument and TorchScript model file as second command
   ! line argument
   num_args = command_argument_count()
   allocate(args(num_args))
   do ix = 1, num_args
      call get_command_argument(ix,args(ix))
   end do

   if (num_args < 1) then
     write(*,*) "Usage: multigpu_infer_fortran <model_file> <device_code> <num_devices>"
     stop 2
   end if

   ! Process device type argument, if provided
   device_code = "cpu"
   if (num_args > 1) then
     device_code = adjustl(trim(args(2)))
   end if
   read(device_code,"(i1)") torch_device

   ! Process num_devices argument, if provided
   num_devices_str = "1"
   if (num_args > 2) then
     num_devices_str = adjustl(trim(args(3)))
   end if
   read(num_devices_str,"(i1)") num_devices

   do device_index = 0, num_devices-1

      ! Initialise data and print the values used
      in_data = [(device_index + i, i = 0, 4)]
      write(unit=stdout, fmt=100) device_index, in_data(:)
      100 format("input on device ", i1,": [", 4(f5.1,","), f5.1,"]")

      ! Create Torch input tensor from the above array and assign it to the first (and only)
      ! element in the array of input tensors.
      ! We use the specified GPU device type with the given device index
      call torch_tensor_from_array(in_tensors(1), in_data, torch_device, device_index=device_index)

      ! Create Torch output tensor from the above array.
      ! Here we use the torch_kCPU device type since the tensor is for output only
      ! i.e. to be subsequently used by Fortran on CPU.
      call torch_tensor_from_array(out_tensors(1), out_data, torch_kCPU)

      ! Load ML model. Ensure that the same device type and device index are used
      ! as for the input data.
      call torch_model_load(model, args(1), torch_device, device_index=device_index)

      ! Infer
      call torch_model_forward(model, in_tensors, out_tensors)

      ! Print the values computed on the current device.
      write(unit=stdout, fmt=200) device_index, out_data(:)
      200 format("output on device ", i1,": [", 4(f5.1,","), f5.1,"]")

      ! Check output tensor matches expected value
      expected = [(2 * (device_index + i), i = 0, 4)]
      test_pass = allclose(out_data, expected, test_name="MultiGPU")

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
