program inference

   ! Import precision info from iso
   use, intrinsic :: iso_fortran_env, only : sp => real32

   ! Import the ml module
   use ml_mod, only : ml_init, ml_routine, ml_final

   implicit none

   ! Set working precision for reals
   integer, parameter :: wp = sp
   integer :: i

   ! Set up Fortran data structures
   real(wp), dimension(5), target :: in_data
   real(wp), dimension(5), target :: out_data
   real(wp), dimension(5), target :: sum_data

   ! Initialise data
   in_data = [0.0, 1.0, 2.0, 3.0, 4.0]
   sum_data(:) = 0.0

   call ml_init()

   ! Loop over ml routine accumulating results
   do i = 1, 10000
      call ml_routine(in_data, out_data)
      sum_data(:) = sum_data(:) + out_data(:)

      in_data(:) = in_data(:) + 1.0
   end do

   call ml_final()

   ! Write out the result of calling the net
   write (*,*) sum_data(:)

end program inference
