program inference

   ! Import precision info from iso
   use, intrinsic :: iso_fortran_env, only : sp => real32

   use, intrinsic :: iso_c_binding, only : c_associated, c_int64_t

   ! Import our library for interfacing with PyTorch
   use ftorch, only : torch_model, torch_tensor, torch_kCPU, torch_kMPS, torch_delete, &
                      torch_tensor_from_array, torch_model_load, torch_model_forward, &
                      operator(+), torch_tensor_print, torch_kFloat32, torch_tensor_empty

   ! Import our tools module for testing utils
   use ftorch_test_utils, only : assert_allclose

   implicit none

   ! Set working precision for reals
   integer, parameter :: wp = sp

   integer :: num_args, ix
   character(len=128), dimension(:), allocatable :: args

   ! Set up Fortran data structures
   real(wp), dimension(5), target :: in_data
   real(wp), dimension(5), target :: out_data
   real(wp), dimension(5), target :: combine_data
   real(wp), dimension(5), target :: expected
   integer, parameter :: tensor_layout(1) = [1]
   integer(c_int64_t), dimension(1), parameter :: tensor_shape = [5]

   ! Set up Torch data structures
   ! The net, a vector of input tensors (in this case we only have one), and the output tensor
   type(torch_model) :: model
   type(torch_tensor), dimension(1) :: in_tensors
   type(torch_tensor), dimension(1) :: out_tensors
   type(torch_tensor) :: combine_tensor

   ! Flag for testing
   logical :: test_pass

   ! Get TorchScript model file as a command line argument
   num_args = command_argument_count()
   allocate(args(num_args))
   do ix = 1, num_args
       call get_command_argument(ix,args(ix))
   end do

   ! Initialise data
   in_data = [0.0_wp, 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp]
   out_data = [0.0_wp, 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp]

   ! Create Torch input/output tensors from the above arrays
   call torch_tensor_from_array(in_tensors(1), in_data, tensor_layout, torch_kCPU)
   call torch_tensor_from_array(out_tensors(1), out_data, tensor_layout, torch_kCPU)

   write(*,*) c_associated(combine_tensor%p)
   call torch_tensor_empty(combine_tensor, 1, tensor_shape, torch_kFloat32, torch_kCPU)
   write(*,*) c_associated(combine_tensor%p)
   call torch_tensor_print(combine_tensor)
   call torch_tensor_print(in_tensors(1))
   call torch_tensor_print(out_tensors(1))
   !call torch_tensor_print(combine_tensor)
   combine_tensor = in_tensors(1) + out_tensors(1)
   write(*,*) c_associated(combine_tensor%p)
   write(*,*) combine_tensor%p
   call torch_tensor_print(combine_tensor)
   write(*,*) c_associated(combine_tensor%p)
   stop

   ! Load ML model
   call torch_model_load(model, args(1), torch_kCPU)

   ! Infer
   call torch_model_forward(model, in_tensors, out_tensors)
   write (*,*) out_data(:)

   ! Check output tensor matches expected value
   expected = [0.0_wp, 2.0_wp, 4.0_wp, 6.0_wp, 8.0_wp]
   test_pass = assert_allclose(out_data, expected, test_name="SimpleNet", rtol=1e-5)

   ! Cleanup
   call torch_delete(model)
   call torch_delete(in_tensors)
   call torch_delete(out_tensors)

   if (.not. test_pass) then
     stop 999
   end if

  write (*,*) "SimpleNet example ran successfully"

end program inference
