program test_tensor
  use, intrinsic :: iso_c_binding, only: c_int64_t, c_float, c_char, c_ptr, c_loc
  use ftorch
  implicit none

  real(kind=8), dimension(:,:), allocatable, target  :: uuu_flattened, vvv_flattened
  real(kind=8), dimension(:,:), allocatable, target    :: lat_reshaped, psfc_reshaped
  real(kind=8), dimension(:,:), allocatable, target  :: gwfcng_x_flattened, gwfcng_y_flattened
  type(torch_tensor), target :: output_tensor
  integer(c_int), parameter :: dims_1D = 2
  integer(c_int), parameter :: dims_2D = 2
  integer(c_int64_t) :: shape_2D_F(dims_2D), shape_2D_C(dims_2D)
  integer(c_int64_t) :: shape_1D_F(dims_1D), shape_1D_C(dims_1D)
  integer(c_int) :: layout_F(dims_1D), layout_C(dims_1D)
  integer :: imax, jmax, kmax, i, j, k

  imax = 1
  jmax = 5
  kmax = 7

  shape_2D_F = (/ kmax, imax*jmax /)
  shape_1D_F = (/ 1, imax*jmax /)
  shape_2D_C = (/ imax*jmax, kmax /)
  shape_1D_C = (/ imax*jmax, 1 /)

  layout_F = (/ 1, 2 /)
  layout_C = (/ 2, 1 /)

  allocate( lat_reshaped(imax*jmax, 1) )
  allocate( uuu_flattened(imax*jmax, kmax) )
  do i = 1, imax*jmax
    lat_reshaped(i, 1) = i
    do k = 1, kmax
      uuu_flattened(i, k) = i + k*100
    end do
  end do

  write(*,*) uuu_flattened

  output_tensor = torch_tensor_from_blob(c_loc(uuu_flattened), &
  dims_2D, shape_2D_C, torch_kFloat64, torch_kCPU, layout_F)

  call torch_tensor_print(output_tensor)

  output_tensor = torch_tensor_from_blob(c_loc(uuu_flattened), &
  dims_2D, shape_2D_F, torch_kFloat64, torch_kCPU, layout_C)

  call torch_tensor_print(output_tensor)

  shape_2D_F = shape(uuu_flattened)
  output_tensor = torch_tensor_from_array_c_double(uuu_flattened, shape_2D_F, torch_kCPU)

  call torch_tensor_print(output_tensor)

  output_tensor = torch_tensor_from_array(uuu_flattened, shape_2D_F, torch_kCPU)
  
  call torch_tensor_print(output_tensor)

  ! output_tensor = torch_tensor_zeros( &
  ! dims_2D, shape_2D_C, torch_kFloat64, torch_kCPU)

  ! call torch_tensor_print(output_tensor)

  ! output_tensor = torch_tensor_ones( &
  ! dims_2D, shape_2D_C, torch_kFloat64, torch_kCPU)

  ! call torch_tensor_print(output_tensor)

end program test_tensor
