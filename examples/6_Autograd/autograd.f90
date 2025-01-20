program example

  ! Import precision info from iso
  use, intrinsic :: iso_fortran_env, only : sp => real32

  ! Import our library for interfacing with PyTorch's Autograd module
  use ftorch, only: assignment(=), operator(+), operator(-), operator(*), operator(/), &
                    operator(**), get_gradient, torch_kCPU, torch_tensor, torch_tensor_backward, &
                    torch_tensor_delete, torch_tensor_from_array, torch_tensor_to_array

  ! Import our tools module for testing utils
  use ftorch_test_utils, only : assert_allclose

  implicit none

  ! Set working precision for reals
  integer, parameter :: wp = sp

  ! Set up Fortran data structures
  integer, parameter :: n = 2
  real(wp), dimension(n), target :: in_data1, in_data2, in_data3
  real(wp), dimension(:), pointer :: out_data1, out_data2, out_data3
  real(wp), dimension(n) :: expected
  integer :: tensor_layout(1) = [1]

  ! Set up Torch data structures
  type(torch_tensor) :: a, b, Q, external_gradient, dQda, dQdb

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
  ! FIXME: Something seems off with gradients related to scalar multiplication and/or division

  ! Extract a Fortran array from a Torch tensor
  call torch_tensor_to_array(Q, out_data1, shape(in_data1))
  write (*,*) "Q = 3 * (a ** 3 - b * b / 2) =", out_data1(:)

  ! Check output tensor matches expected value
  expected(:) = [-12.0_wp, 65.0_wp]
  if (.not. assert_allclose(out_data1, expected, test_name="autograd_Q")) then
    call clean_up()
    print *, "Error :: value of Q does not match expected value"
    stop 999
  end if

  ! Back-propagation
  in_data3(:) = [1.0_wp, 1.0_wp]
  call torch_tensor_from_array(external_gradient, in_data3, tensor_layout, torch_kCPU)
  call torch_tensor_backward(Q, external_gradient)
  dQda = get_gradient(a)
  dQdb = get_gradient(b)

  ! Extract Fortran arrays from the Torch tensors and check the gradients take expected values
  call torch_tensor_to_array(dQda, out_data2, shape(in_data1))
  print *, "dQda", out_data2
  expected(:) = [36.0_wp, 81.0_wp]
  if (.not. assert_allclose(out_data2, expected, test_name="autograd_dQdb")) then
    call clean_up()
    print *, "Error :: value of dQdb does not match expected value"
    stop 999
  end if
  call torch_tensor_to_array(dQdb, out_data3, shape(in_data1))
  print *, "dQdb", out_data3
  expected(:) = [-12.0_wp, -8.0_wp]
  if (.not. assert_allclose(out_data3, expected, test_name="autograd_dQdb")) then
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
      call torch_tensor_delete(Q)
      call torch_tensor_delete(external_gradient)
    end subroutine clean_up

end program example
