module precisn
   use, intrinsic :: iso_c_binding, only: c_float
   implicit none
   private
   integer, parameter:: wp = c_float
   public wp
end module precisn

program inference

   ! Imports primitives used to interface with C
   use, intrinsic :: iso_c_binding, only: c_int64_t, c_null_char, c_loc
   ! Import our library for interfacing with PyTorch
   use ftorch
   use precisn, only: wp

   implicit none

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

      real(wp), dimension(:,:,:,:), allocatable, target :: in_data
      integer(c_int), parameter :: n_inputs = 1
      real(wp), dimension(:,:), allocatable, target :: out_data

      integer(c_int), parameter :: in_dims = 4
      integer(c_int64_t) :: in_shape(in_dims) = [1, 3, 224, 224]
      integer(c_int) :: in_layout(in_dims) = [1,2,3,4]
      integer(c_int), parameter :: out_dims = 2
      integer(c_int64_t) :: out_shape(out_dims) = [1, 1000]
      integer(c_int) :: out_layout(out_dims) = [1,2]

      ! File containing input tensor binary
      character(len=*), parameter :: filename = '../image_tensor.dat'
      ! Length of tensor
      integer, parameter :: N = 150528

      ! Get TorchScript model file as a command line argument
      num_args = command_argument_count()
      allocate(args(num_args))
      do ix = 1, num_args
         call get_command_argument(ix,args(ix))
      end do

      ! Allocate one-dimensional input/output arrays, based on multiplication of all input/output dimension sizes
      allocate(in_data(in_shape(1), in_shape(2), in_shape(3), in_shape(4)))
      allocate(out_data(out_shape(1), out_shape(2)))

      call load_data(filename, N, in_data, in_dims, in_shape)

      ! Create input/output tensors from the above arrays
      in_tensor(1) = torch_tensor_from_blob(c_loc(in_data), in_dims, in_shape, torch_kFloat32, torch_kCPU, in_layout)
      out_tensor = torch_tensor_from_blob(c_loc(out_data), out_dims, out_shape, torch_kFloat32, torch_kCPU, out_layout)

      ! Load ML model (edit this line to use different models)
      model = torch_module_load(trim(args(1))//c_null_char)

      ! Infer
      call torch_module_forward(model, in_tensor, n_inputs, out_tensor)

      ! Output results
      write (*,*) "Max output: ", maxval(out_data)
      write (*,*) "Index: ", maxloc(out_data)

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
      real(wp), dimension(:,:,:,:), allocatable, target, intent(inout) :: in_data

      integer(c_int), intent(in) :: in_dims
      integer(c_int64_t), intent(in) :: in_shape(in_dims)

      real(wp) :: flat_data(N)
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

      ! Initialise data
      count = 1
      do idx_1 = 1, in_shape(1)
         do idx_2 = 1, in_shape(2)
            do idx_3 = 1, in_shape(3)
               do idx_4 = 1, in_shape(4)
                  in_data(idx_1, idx_2, idx_3, idx_4) = flat_data(count)
                  count = count + 1
               end do
            end do
         end do
      end do

   end subroutine load_data

end program inference
