program example

  ! Import precision info from iso
  use, intrinsic :: iso_fortran_env, only : sp => real32

  ! Import our library for interfacing with PyTorch's Autograd module
  use ftorch, only: assignment(=), operator(+), operator(-), operator(*), operator(/), &
                    operator(**), torch_kCPU, torch_tensor, torch_tensor_backward, &
                    torch_tensor_delete, torch_tensor_from_array, torch_tensor_to_array

  ! Import our tools module for testing utils
  use ftorch_test_utils, only : assert_allclose

  implicit none

  ! Set working precision for reals
  integer, parameter :: wp = sp

  ! Set up Fortran data structures
  integer, parameter :: n = 2
  real(wp), dimension(n), target :: in_data1, in_data2, in_data3
  real(wp), dimension(:), pointer :: out_data
  real(wp), dimension(n) :: expected
  integer :: tensor_layout(1) = [1]

  ! Set up Torch data structures
  type(torch_tensor) :: a, b, Q, external_gradient

  ! Initialise input arrays as in Python example
  in_data1(:) = [2.0_wp, 3.0_wp]
  in_data2(:) = [6.0_wp, 4.0_wp]

  ! Construct a Torch Tensor from a Fortran array
  call torch_tensor_from_array(a, in_data1, tensor_layout, torch_kCPU, requires_grad=.true.)
  call torch_tensor_from_array(b, in_data2, tensor_layout, torch_kCPU, requires_grad=.true.)

  ! Check arithmetic operations work for torch_tensors
  write (*,*) "a = ", in_data1(:)
  write (*,*) "b = ", in_data2(:)
  Q = 3 * (a**3 - b * b / 3)

  ! Extract a Fortran array from a Torch tensor
  call torch_tensor_to_array(Q, out_data, shape(in_data1))
  write (*,*) "Q = 3 * (a ** 3 - b * b / 2) =", out_data(:)

  ! Check output tensor matches expected value
  expected(:) = [-12.0_wp, 65.0_wp]
  if (.not. assert_allclose(out_data, expected, test_name="torch_tensor_to_array")) then
    call clean_up()
    print *, "Error :: out_data does not match expected value"
    stop 999
  end if

  ! Back-propagation
  in_data3(:) = [1.0_wp, 1.0_wp]
  call torch_tensor_from_array(external_gradient, in_data3, tensor_layout, torch_kCPU)
  call torch_tensor_backward(Q, external_gradient)

  call clean_up()
  write (*,*) "Autograd example ran successfully"

  contains

    ! Subroutine for freeing memory and nullifying pointers used in the example
    subroutine clean_up()
      nullify(out_data)
      call torch_tensor_delete(a)
      call torch_tensor_delete(b)
      call torch_tensor_delete(Q)
      call torch_tensor_delete(external_gradient)
    end subroutine clean_up

end program example
