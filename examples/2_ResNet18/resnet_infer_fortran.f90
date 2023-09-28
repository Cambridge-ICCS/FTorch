program inference

   ! Imports primitives used to interface with C
   use, intrinsic :: iso_c_binding, only: c_sp=>c_float, c_dp=>c_double, c_int64_t, c_null_char, c_loc
   use, intrinsic :: iso_fortran_env, only : sp => real32, dp => real64
   ! Import our library for interfacing with PyTorch
   use :: ftorch

   implicit none

   ! Define working precision for C primitives
   ! Precision must match `wp` in resnet18.py and `wp_torch` in pt2ts.py
   integer, parameter :: c_wp = c_sp
   integer, parameter :: wp = sp
   integer, parameter :: torch_wp = torch_kFloat32

   call main()

contains

   subroutine main()

      implicit none

      integer :: num_args, ix
      character(len=128), dimension(:), allocatable :: args

      ! Set up types of input and output data and the interface with C
      type(torch_module) :: model
      type(torch_tensor), dimension(1) :: in_tensor
      type(torch_tensor) :: out_tensor

      real(c_wp), dimension(:,:,:,:), allocatable, target :: in_data
      integer(c_int), parameter :: n_inputs = 1
      real(c_wp), dimension(:,:), allocatable, target :: out_data

      integer(c_int), parameter :: in_dims = 4
      integer(c_int64_t) :: in_shape(in_dims) = [1, 3, 224, 224]
      integer(c_int) :: in_layout(in_dims) = [1,2,3,4]
      integer(c_int), parameter :: out_dims = 2
      integer(c_int64_t) :: out_shape(out_dims) = [1, 1000]
      integer(c_int) :: out_layout(out_dims) = [1,2]

      ! Binary file containing input tensor
      character(len=*), parameter :: filename = '../data/image_tensor.dat'
      ! Text file containing categories
      character(len=*), parameter :: filename_cats = '../data/categories.txt'

      ! Length of tensor and number of categories
      integer, parameter :: N = 150528
      integer, parameter :: N_cats = 1000

      ! Outputs
      integer :: index(2)
      real(wp), dimension(:,:), allocatable, target :: probabilities
      real(wp), parameter :: expected_prob = 0.8846225142478943
      character(len=100) :: categories(N_cats)
      real(wp) :: probability

      ! Get TorchScript model file as a command line argument
      num_args = command_argument_count()
      allocate(args(num_args))
      do ix = 1, num_args
         call get_command_argument(ix,args(ix))
      end do

      ! Allocate one-dimensional input/output arrays, based on multiplication of all input/output dimension sizes
      allocate(in_data(in_shape(1), in_shape(2), in_shape(3), in_shape(4)))
      allocate(out_data(out_shape(1), out_shape(2)))
      allocate(probabilities(out_shape(1), out_shape(2)))

      call load_data(filename, N, in_data, in_dims, in_shape)

      ! Create input/output tensors from the above arrays
      in_tensor(1) = torch_tensor_from_blob(c_loc(in_data), in_dims, in_shape, torch_wp, torch_kCPU, in_layout)
      out_tensor = torch_tensor_from_blob(c_loc(out_data), out_dims, out_shape, torch_wp, torch_kCPU, out_layout)

      ! Load ML model (edit this line to use different models)
      model = torch_module_load(trim(args(1))//c_null_char)

      ! Infer
      call torch_module_forward(model, in_tensor, n_inputs, out_tensor)

      ! Load categories
      call load_categories(filename_cats, N_cats, categories)

      ! Calculate probabilities and output results
      call calc_probs(out_data, probabilities, out_dims, out_shape)
      index = maxloc(probabilities)
      probability = maxval(probabilities)

      ! Check top probability matches expected value
      call assert_real(probability, expected_prob, test_name="Check probability", rtol_opt=1e-5)

      write (*,*) "Top result"
      write (*,*) ""
      write (*,*) trim(categories(index(2))), " (id=", index(2), "), : probability =", probability

      ! Cleanup
      call torch_module_delete(model)
      call torch_tensor_delete(in_tensor(1))
      call torch_tensor_delete(out_tensor)
      deallocate(in_data)
      deallocate(out_data)

   end subroutine main

   subroutine load_data(filename, N, in_data, in_dims, in_shape)

      implicit none

      character(len=*), intent(in) :: filename
      integer, intent(in) :: N
      real(c_wp), dimension(:,:,:,:), allocatable, target, intent(inout) :: in_data

      integer(c_int), intent(in) :: in_dims
      integer(c_int64_t), intent(in) :: in_shape(in_dims)

      real(c_wp) :: flat_data(N)
      integer :: ios, count, idx_1, idx_2, idx_3, idx_4
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
      character(len=100), intent(inout) :: categories(N_cats)

      integer :: ios, i
      character(len=100) :: ioerrmsg

      open (unit=11, file=filename_cats, form='formatted', access='stream', action='read', iostat=ios, iomsg=ioerrmsg)
      if (ios /= 0) then
        print *, ioerrmsg
        stop 1
      end if

      read(11, '(a)') categories
      close(11)

   end subroutine load_categories

   subroutine calc_probs(out_data, probabilities, out_dims, out_shape)

      implicit none

      integer(c_int), intent(in) :: out_dims
      integer(c_int64_t), intent(in) :: out_shape(out_dims)
      real(c_wp), dimension(:,:), allocatable, target, intent(in) :: out_data
      real(wp), dimension(:,:), allocatable, target, intent(inout) :: probabilities
      real(wp) :: prob_sum
      integer :: i, j

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
