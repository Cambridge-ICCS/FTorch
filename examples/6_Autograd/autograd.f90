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
  real(wp), dimension(:), pointer :: out_data1, out_data2, out_data3
  integer :: tensor_layout(1) = [1]

  ! Set up Torch data structures
  type(torch_tensor) :: a, b, Q, dQda, dQdb, multiplier, divisor

  ! Initialise input arrays as in the Python example and construct Torch Tensors from them
  call torch_tensor_from_array(a, [2.0_wp, 3.0_wp], tensor_layout, torch_kCPU, requires_grad=.true.)
  call torch_tensor_from_array(b, [6.0_wp, 4.0_wp], tensor_layout, torch_kCPU, requires_grad=.true.)

  ! Scalar multiplication and division are not currently implemented in FTorch. However, you can
  ! achieve the same thing by defining a rank-1 tensor with a single entry, as follows:
  call torch_tensor_from_array(multiplier, [3.0_wp], tensor_layout, torch_kCPU)
  call torch_tensor_from_array(divisor, [3.0_wp], tensor_layout, torch_kCPU)

  ! Compute the same mathematical expression as in the Python example
  Q = multiplier * (a**3 - b * b / divisor)

  ! Extract a Fortran array from the Torch tensor
  call torch_tensor_to_array(Q, out_data1, [2])
  write (*,*) "Q = 3 * (a^3 - b*b/3) = 3*a^3 - b^2 = ", out_data1(:)

  ! Check output tensor matches expected value
  if (.not. assert_allclose(out_data1, [-12.0_wp, 65.0_wp], test_name="autograd_Q")) then
    call clean_up()
    print *, "Error :: value of Q does not match expected value"
    stop 999
  end if

  ! Run the back-propagation operator
  call torch_tensor_backward(Q)
  dQda = a%grad()
  dQdb = b%grad()

  ! Extract Fortran arrays from the Torch tensors and check the gradients take expected values
  call torch_tensor_to_array(dQda, out_data2, [2])
  print *, "dQda = 9*a^2 = ", out_data2
  if (.not. assert_allclose(out_data2, [36.0_wp, 81.0_wp], test_name="autograd_dQdb")) then
    call clean_up()
    print *, "Error :: value of dQdb does not match expected value"
    stop 999
  end if
  call torch_tensor_to_array(dQdb, out_data3, [2])
  print *, "dQdb = - 2*b = ", out_data3
  if (.not. assert_allclose(out_data3, [-12.0_wp, -8.0_wp], test_name="autograd_dQdb")) then
    call clean_up()
    print *, "Error :: value of dQdb does not match expected value"
    stop 999
  end if

  call clean_up()
  write (*,*) "Autograd example ran successfully"

  contains

    ! Subroutine for freeing memory and nullifying pointers used in the example
    subroutine clean_up()
      nullify(out_data1)
      nullify(out_data2)
      nullify(out_data3)
      call torch_tensor_delete(a)
      call torch_tensor_delete(b)
      call torch_tensor_delete(multiplier)
      call torch_tensor_delete(divisor)
      call torch_tensor_delete(Q)
    end subroutine clean_up

end program example
