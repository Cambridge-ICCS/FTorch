program inference

   use, intrinsic :: iso_fortran_env, only : sp => real32
   ! Import our library for interfacing with PyTorch
   use :: ftorch

   implicit none

   integer, parameter :: wp = sp

   call main()

contains

   subroutine main()

      implicit none

      integer :: num_args, ix
      character(len=128), dimension(:), allocatable :: args

      ! Set up types of input and output data
      type(torch_model) :: model
      type(torch_tensor), dimension(1) :: in_tensors
      type(torch_tensor), dimension(1) :: out_tensors

      real(wp), dimension(:,:,:,:), allocatable, target :: in_data
      real(wp), dimension(:,:), allocatable, target :: out_data

      integer, parameter :: in_dims = 4
      integer :: in_shape(in_dims) = [1, 3, 224, 224]
      integer :: in_layout(in_dims) = [1, 2, 3, 4]
      integer, parameter :: out_dims = 2
      integer :: out_shape(out_dims) = [1, 1000]
      integer :: out_layout(out_dims) = [1, 2]

      ! Path to input data
      character(len=100) :: data_dir
      ! Binary file containing input tensor
      character(len=116) :: filename
      ! Text file containing categories
      character(len=114) :: filename_cats

      ! Length of tensor and number of categories
      integer, parameter :: tensor_length = 150528
      integer, parameter :: N_cats = 1000

      ! Outputs
      integer :: idx(2)
      real(wp), dimension(:,:), allocatable :: probabilities
      real(wp), parameter :: expected_prob = 0.8846225142478943
      character(len=100) :: categories(N_cats)
      real(wp) :: probability

      ! Get TorchScript model file as a command line argument
      num_args = command_argument_count()
      allocate(args(num_args))
      do ix = 1, num_args
         call get_command_argument(ix,args(ix))
      end do

      ! Process data directory argument, if provided
      if (num_args > 1) then
        data_dir = args(2)
      else
        data_dir = "../data"
      end if
      filename = trim(data_dir)//"/image_tensor.dat"
      filename_cats =  trim(data_dir)//"/categories.txt"

      ! Allocate one-dimensional input/output arrays, based on multiplication of all input/output dimension sizes
      allocate(in_data(in_shape(1), in_shape(2), in_shape(3), in_shape(4)))
      allocate(out_data(out_shape(1), out_shape(2)))
      allocate(probabilities(out_shape(1), out_shape(2)))

      call load_data(filename, tensor_length, in_data)

      ! Create input/output tensors from the above arrays
      call torch_tensor_from_array(in_tensors(1), in_data, in_layout, torch_kCPU)

      call torch_tensor_from_array(out_tensors(1), out_data, out_layout, torch_kCPU)

      ! Load ML model (edit this line to use different models)
      call torch_model_load(model, args(1))

      ! Infer
      call torch_model_forward(model, in_tensors, out_tensors)

      ! Load categories
      call load_categories(filename_cats, N_cats, categories)

      ! Calculate probabilities and output results
      call calc_probs(out_data, probabilities)
      idx = maxloc(probabilities)
      probability = maxval(probabilities)

      ! Check top probability matches expected value
      call assert_real(probability, expected_prob, test_name="Check probability", rtol_opt=1e-5)

      write (*,*) "Top result"
      write (*,*) ""
      write (*,*) trim(categories(idx(2))), " (id=", idx(2), "), : probability =", probability

      ! Cleanup
      call torch_model_delete(model)
      call torch_tensor_delete(in_tensors(1))
      call torch_tensor_delete(out_tensors(1))
      deallocate(in_data)
      deallocate(out_data)
      deallocate(probabilities)
      deallocate(args)

   end subroutine main

   subroutine load_data(filename, tensor_length, in_data)

      implicit none

      character(len=*), intent(in) :: filename
      integer, intent(in) :: tensor_length
      real(wp), dimension(:,:,:,:), intent(out) :: in_data

      real(wp) :: flat_data(tensor_length)
      integer :: ios
      character(len=100) :: ioerrmsg

      ! Read input tensor from Python script
      open(unit=10, file=filename, status='old', access='stream', form='unformatted', action="read", iostat=ios, iomsg=ioerrmsg)
      if (ios /= 0) then
      print *, ioerrmsg
      stop 1
      end if

      read(10, iostat=ios, iomsg=ioerrmsg) flat_data
      if (ios /= 0) then
         print *, ioerrmsg
         stop 1
      end if

      close(10)

      ! Reshape data to tensor input shape
      ! This assumes the data from Python was transposed before saving
      in_data = reshape(flat_data, shape(in_data))

   end subroutine load_data

   subroutine load_categories(filename_cats, N_cats, categories)

      implicit none

      character(len=*), intent(in) :: filename_cats
      integer, intent(in) :: N_cats
      character(len=100), intent(out) :: categories(N_cats)

      integer :: ios
      character(len=100) :: ioerrmsg

      open (unit=11, file=filename_cats, form='formatted', access='stream', action='read', iostat=ios, iomsg=ioerrmsg)
      if (ios /= 0) then
        print *, ioerrmsg
        stop 1
      end if

      read(11, '(a)') categories
      close(11)

   end subroutine load_categories

   subroutine calc_probs(out_data, probabilities)

      implicit none

      real(wp), dimension(:,:), intent(in) :: out_data
      real(wp), dimension(:,:), intent(out) :: probabilities
      real(wp) :: prob_sum

      ! Apply softmax function to calculate probabilties
      probabilities = exp(out_data)
      prob_sum = sum(probabilities)
      probabilities = probabilities / prob_sum

   end subroutine calc_probs

   subroutine assert_real(a, b, test_name, rtol_opt)

      implicit none

      character(len=*) :: test_name
      real, intent(in) :: a, b
      real, optional :: rtol_opt
      real :: relative_error, rtol

      character(len=15) :: pass, fail

      fail = char(27)//'[31m'//'FAILED'//char(27)//'[0m'
      pass = char(27)//'[32m'//'PASSED'//char(27)//'[0m'

      if (.not. present(rtol_opt)) then
        rtol = 1e-5
      else
         rtol = rtol_opt
      end if

      relative_error = abs(a/b - 1.)

      if (relative_error > rtol) then
        write(*, '(A, " :: [", A, "] maximum relative error = ", E11.4)') fail, trim(test_name), relative_error
      else
        write(*, '(A, " :: [", A, "] maximum relative error = ", E11.4)') pass, trim(test_name), relative_error
      end if

   end subroutine assert_real

end program inference
