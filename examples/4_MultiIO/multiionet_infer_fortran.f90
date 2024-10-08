program inference

   ! Import precision info from iso
   use, intrinsic :: iso_fortran_env, only : sp => real32

   ! Import our library for interfacing with PyTorch
   use ftorch

   ! Import our tools module for testing utils
   use utils, only : assert_real_array_1d

   implicit none
  
   ! Set working precision for reals
   integer, parameter :: wp = sp
   
   integer :: num_args, ix
   character(len=128), dimension(:), allocatable :: args

   ! Set up Fortran data structures
   real(wp), dimension(4), target :: in_data1
   real(wp), dimension(4), target :: in_data2
   real(wp), dimension(4), target :: out_data1
   real(wp), dimension(4), target :: out_data2
   real(wp), dimension(4) :: expected
   integer :: tensor_layout(1) = [1]

   ! Set up Torch data structures
   ! The net, a vector of input tensors (in this case we only have one), and the output tensor
   type(torch_model) :: model
   type(torch_tensor), dimension(2) :: in_tensors
   type(torch_tensor), dimension(2) :: out_tensors

   ! Flag for testing
   logical :: test_pass

   ! Get TorchScript model file as a command line argument
   num_args = command_argument_count()
   allocate(args(num_args))
   do ix = 1, num_args
       call get_command_argument(ix,args(ix))
   end do

   ! Initialise data
   in_data1(:) = [0.0, 1.0, 2.0, 3.0]
   in_data2(:) = [0.0, -1.0, -2.0, -3.0]

   ! Create Torch input/output tensors from the above arrays
   call torch_tensor_from_array(in_tensors(1), in_data1, tensor_layout, torch_kCPU)
   call torch_tensor_from_array(in_tensors(2), in_data2, tensor_layout, torch_kCPU)
   call torch_tensor_from_array(out_tensors(1), out_data1, tensor_layout, torch_kCPU)
   call torch_tensor_from_array(out_tensors(2), out_data2, tensor_layout, torch_kCPU)

   ! Load ML model
   call torch_model_load(model, args(1))

   ! Infer
   call torch_model_forward(model, in_tensors, out_tensors)
   write (*,*) out_data1
   write (*,*) out_data2

   ! Check output tensors match expected values
   expected = [0.0, 2.0, 4.0, 6.0]
   test_pass = assert_real_array_1d(out_data1, expected, test_name="MultiIO array1", rtol=1e-5)
   expected = [0.0, -3.0, -6.0, -9.0]
   test_pass = assert_real_array_1d(out_data2, expected, test_name="MultiIO array2", rtol=1e-5)

   ! Cleanup
   call torch_delete(model)
   call torch_delete(in_tensors)
   call torch_delete(out_tensors)

   if (.not. test_pass) then
     stop 999
   end if

end program inference
