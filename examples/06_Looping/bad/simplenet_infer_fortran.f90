program inference

   ! Import precision info from iso
   use, intrinsic :: iso_fortran_env, only : sp => real32

   ! Import the ml module
   use ml_mod, only : ml_routine

   implicit none

   ! Set working precision for reals
   integer, parameter :: wp = sp
   integer :: i

   ! Set up Fortran data structures
   real(wp), dimension(5), target :: in_data
   real(wp), dimension(5), target :: out_data
   real(wp), dimension(5), target :: sum_data

   ! Initialise data
   in_data = [0.0_wp, 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp]
   sum_data(:) = 0.0_wp

   ! Loop over ml routine accumulating results
   do i = 1, 10000
      call ml_routine(in_data, out_data)
      sum_data(:) = sum_data(:) + out_data(:)

      in_data(:) = in_data(:) + 1.0_wp
   end do

   ! Write out the result of calling the net
   write (*,*) sum_data(:)

end program inference
