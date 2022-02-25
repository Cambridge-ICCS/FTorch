program inference

   use, intrinsic :: iso_fortran_env, only: int32, int64, real32
   use, intrinsic :: iso_c_binding, only: c_int, c_float, c_char, c_null_char, c_ptr, c_loc
   use mod_torch

   implicit none

   character(len=:), allocatable :: filename
   integer(int32), parameter :: dims = 4
   integer(int64) :: shape(dims) = [1, 3, 224, 224]
   type(c_ptr) :: model, input, output, input_data
   real(c_float), allocatable, target :: data(:)

   allocate(data(product(shape)))
   data = 1.0d0
   input_data = c_loc(data)

   ! Create input tensor
   input = torch_from_blob(input_data, dims, shape, 6_c_int, 0_c_int)
   ! Load ML model
   model = torch_jit_load(c_char_"../annotated_cpu.pt"//c_null_char)
   ! Deploy
   output = torch_jit_module_forward(model, input)

   ! Cleanup
   call torch_jit_module_delete(model)
   call torch_tensor_delete(input)
   call torch_tensor_delete(output)
   deallocate(data)

contains

   subroutine alloc(a, n)
      real, allocatable, intent(inout) :: a(:)
      integer, intent(in) :: n
      integer :: stat
      character(100) :: errmsg

      if (allocated(a)) call free(a)
      allocate(a(n), stat=stat, errmsg=errmsg)
      if (stat > 0) error stop errmsg
   end subroutine alloc

   subroutine free(a)
      real, allocatable, intent(inout) :: a(:)
      integer :: stat
      character(100) :: errmsg

      if (.not. allocated(a)) return
      deallocate(a, stat=stat, errmsg=errmsg)
      if (stat > 0) error stop errmsg
   end subroutine free

end program inference
