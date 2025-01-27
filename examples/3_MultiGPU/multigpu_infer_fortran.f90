program inference

   ! Import precision info from iso
   use, intrinsic :: iso_fortran_env, only : sp => real32

   ! Import our library for interfacing with PyTorch
   use ftorch, only : torch_model, torch_tensor, torch_kCUDA, torch_kCPU, &
                      torch_tensor_from_array, torch_model_load, torch_model_forward, &
                      torch_delete

   implicit none

   ! Set precision for reals
   integer, parameter :: wp = sp

   integer :: num_args, ix
   character(len=128), dimension(:), allocatable :: args

   ! Set up Fortran data structures
   real(wp), dimension(5), target :: in_data
   real(wp), dimension(5), target :: out_data
   integer, parameter :: tensor_layout(1) = [1]

   ! Set up Torch data structures
   type(torch_model) :: model
   type(torch_tensor), dimension(1) :: in_tensors
   type(torch_tensor), dimension(1) :: out_tensors

   integer, parameter :: num_devices = 2
   integer :: device_index, i

   ! Get TorchScript model file as a command line argument
   num_args = command_argument_count()
   allocate(args(num_args))
   do ix = 1, num_args
      call get_command_argument(ix,args(ix))
   end do

   do device_index = 0, num_devices-1

      ! Initialise data and print the values used
      in_data = [(device_index + i, i = 0, 4)]
      write (6, 100) device_index, in_data(:)
      100 format("input on device ", i1,": [", 4(f5.1,","), f5.1,"]")

      ! Create Torch input tensor from the above array and assign it to the first (and only)
      ! element in the array of input tensors.
      ! We use the torch_kCUDA device type with the given device index
      call torch_tensor_from_array(in_tensors(1), in_data, tensor_layout, torch_kCUDA, &
                                   device_index=device_index)

      ! Create Torch output tensor from the above array.
      ! Here we use the torch_kCPU device type since the tensor is for output only
      ! i.e. to be subsequently used by Fortran on CPU.
      call torch_tensor_from_array(out_tensors(1), out_data, tensor_layout, torch_kCPU)

      ! Load ML model. Ensure that the same device type and device index are used
      ! as for the input data.
      call torch_model_load(model, args(1), torch_kCUDA, device_index=device_index)

      ! Infer
      call torch_model_forward(model, in_tensors, out_tensors)

      ! Print the values computed on the current device.
      write (6, 200) device_index, out_data(:)
      200 format("output on device ", i1,": [", 4(f5.1,","), f5.1,"]")

      ! Cleanup
      call torch_delete(model)
      call torch_delete(in_tensors)
      call torch_delete(out_tensors)

   end do

   write (*,*) "MultiGPU example ran successfully"

end program inference
