program example

  ! Import precision info from iso
  use, intrinsic :: iso_fortran_env, only : sp => real32

  ! Import our library for interfacing with PyTorch's Autograd module
  use ftorch, only: assignment(=), operator(+), operator(-), operator(*), &
    operator(/), operator(**), torch_kCPU, torch_tensor, torch_tensor_delete, &
    torch_tensor_from_array, torch_tensor_to_array

  ! Import our tools module for testing utils
  use ftorch_test_utils, only : assert_allclose

  implicit none

  ! Set working precision for reals
  integer, parameter :: wp = sp

  ! Set up Fortran data structures
  integer, parameter :: n=2, m=1
  real(wp), dimension(n,m), target :: in_data1
  real(wp), dimension(n,m), target :: in_data2
  real(wp), dimension(:,:), pointer :: out_data
  real(wp), dimension(n,m) :: expected
  integer :: tensor_layout(2) = [1, 2]

  ! Flag for testing
  logical :: test_pass

  ! Set up Torch data structures
  type(torch_tensor) :: a, b, Q

  ! Initialise input arrays as in Python example
  in_data1(:,1) = [2.0_wp, 3.0_wp]
  in_data2(:,1) = [6.0_wp, 4.0_wp]

  ! Construct a Torch Tensor from a Fortran array
  ! TODO: Implement requires_grad=.true.
  call torch_tensor_from_array(a, in_data1, tensor_layout, torch_kCPU)
  call torch_tensor_from_array(b, in_data2, tensor_layout, torch_kCPU)

  ! check tensor rank and shape match those of in_data
  if ((a%get_rank() /= 2) .or. (b%get_rank() /= 2)) then
    print *, "Error :: rank should be 2"
    stop 1
  end if
  if (any(a%get_shape() /= [n, m]) .or. any(b%get_shape() /= [n, m])) then
    write(6,"('Error :: shape should be (',i1,', ',i1,')')") n, m
    stop 1
  end if

  ! Check arithmetic operations work for torch_tensors
  write (*,*) "a = ", in_data1(:,1)
  write (*,*) "b = ", in_data2(:,1)
  Q = 3 * (a**3 - b * b / 3)

  ! Extract a Fortran array from a Torch tensor
  call torch_tensor_to_array(Q, out_data, shape(in_data1))
  write (*,*) "Q = 3 * (a ** 3 - b * b / 2) =", out_data(:,1)

  ! Check output tensor matches expected value
  expected(:,1) = [-12.0_wp, 65.0_wp]
  test_pass = assert_allclose(out_data, expected, test_name="torch_tensor_to_array", rtol=1e-5)
  if (.not. test_pass) then
    call clean_up()
    print *, "Error :: out_data does not match expected value"
    stop 999
  end if

  ! Check first input array is unchanged by the arithmetic operations
  expected(:,1) = [2.0_wp, 3.0_wp]
  test_pass = assert_allclose(in_data1, expected, test_name="torch_tensor_to_array", rtol=1e-5)
  if (.not. test_pass) then
    call clean_up()
    print *, "Error :: in_data1 was changed during arithmetic operations"
    stop 999
  end if

  ! Check second input array is unchanged by the arithmetic operations
  expected(:,1) = [6.0_wp, 4.0_wp]
  test_pass = assert_allclose(in_data2, expected, test_name="torch_tensor_to_array", rtol=1e-5)
  if (.not. test_pass) then
    call clean_up()
    print *, "Error :: in_data2 was changed during arithmetic operations"
    stop 999
  end if

  ! Back-propagation
  ! TODO: Requires API extension

  ! Cleanup
  call clean_up()
  write (*,*) "Autograd example ran successfully"

  contains

    ! Subroutine for freeing memory and nullifying pointers used in the example
    subroutine clean_up()
      nullify(out_data)
      call torch_tensor_delete(a)
      call torch_tensor_delete(b)
      call torch_tensor_delete(Q)
    end subroutine clean_up

end program example
