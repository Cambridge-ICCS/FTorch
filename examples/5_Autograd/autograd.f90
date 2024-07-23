program example

  ! Import precision info from iso
  use, intrinsic :: iso_fortran_env, only : sp => real32

  ! Import our library for interfacing with PyTorch's Autograd module
  use ftorch

  implicit none

  ! Set working precision for reals
  integer, parameter :: wp = sp

  ! Set up Fortran data structures
  real(wp), dimension(2), target :: in_data
  real(wp), dimension(:), pointer :: out_data
  integer :: tensor_layout(1) = [1]

  ! Set up Torch data structures
  type(torch_tensor) :: a

  ! Construct a Torch Tensor from a Fortran array
  in_data(:) = [2.0, 3.0]
  call torch_tensor_from_array(a, in_data, tensor_layout, torch_kCPU)

  ! Extract a Fortran array from a Torch tensor
  allocate(out_data(2))
  call torch_tensor_to_array(a, out_data)
  write (*,*) "a = ", out_data(:)

  ! Cleanup
  nullify(out_data)
  call torch_tensor_delete(a)

end program example
