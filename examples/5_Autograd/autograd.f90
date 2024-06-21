program example

  ! Import precision info from iso
  use, intrinsic :: iso_fortran_env, only : sp => real32

  ! Import our library for interfacing with PyTorch's Autograd module
  use ftorch

  implicit none

  ! Set working precision for reals
  integer, parameter :: wp = sp

   ! Set up Fortran data structures
  real(wp), dimension(2), target :: in_data1
  real(wp), dimension(2), target :: in_data2
  real(wp), dimension(2), target :: out_data
  integer :: tensor_layout(1) = [1]

   ! Set up Torch data structures
  type(torch_tensor) :: a, b, Q

   ! Initialise data
  in_data1(:) = [2.0, 3.0]
  in_data2(:) = [6.0, 4.0]

  ! FIXME: requires_grad=.true.
  call torch_tensor_from_array(a, in_data1, tensor_layout, torch_kCPU)
  call torch_tensor_from_array(b, in_data2, tensor_layout, torch_kCPU)
  call torch_tensor_from_array(Q, out_data, tensor_layout, torch_kCPU)

  ! Check arithmetic operations work for torch_tensors
  write (*,*) "a = ", in_data1(:)
  write (*,*) "b = ", in_data2(:)
  Q = 3 * a ** 3 - b ** 2
  write (*,*) "Q = 3 * a ** 3 - b ** 2 =", out_data(:)

  ! Check a and b are unchanged by the arithmetic operations
  write (*,*) "a = ", in_data1(:)
  write (*,*) "b = ", in_data2(:)

  ! TODO: Backward
  !   Requires API extension

  ! Cleanup
  call torch_tensor_delete(a)
  call torch_tensor_delete(b)
  call torch_tensor_delete(Q)

end program example
