!| Unit tests for FTorch's tensor manipulation functionality.
!
!  * License
!    FTorch is released under an MIT license.
!    See the [LICENSE](https://github.com/Cambridge-ICCS/FTorch/blob/main/LICENSE)
!    file for details.
module unittest_tensor_manipulation
  use testdrive, only: unittest_type, new_unittest, error_type, check
  use ftorch, only: assignment(=), torch_kCPU, torch_kFloat32, torch_tensor, torch_tensor_from_array
  use ftorch_test_utils, only: assert_allclose

  use, intrinsic :: iso_c_binding, only : c_int64_t

  implicit none

  private
  public :: collect_tensor_manipulation_suite

  integer, parameter :: device_type = torch_kCPU

contains

  !> Collect all exported unit tests
  subroutine collect_tensor_manipulation_suite(testsuite)
    type(unittest_type), allocatable, intent(out) :: testsuite(:)

    testsuite = [ &
      new_unittest("valid", test_torch_tensor_zero) &
    ]
  end subroutine collect_tensor_manipulation_suite

  subroutine test_torch_tensor_zero(error)
    use, intrinsic :: iso_fortran_env, only: sp => real32

    ! Set working precision for reals
    integer, parameter :: wp = sp

    type(error_type), allocatable, intent(out) :: error
    type(torch_tensor) :: tensor
    integer, parameter :: ndims = 2
    integer, parameter :: dtype = torch_kFloat32
    real(wp), dimension(2,3), target :: in_data
    real(wp), dimension(2,3) :: expected
    logical :: test_pass

    ! Create an arbitrary input array
    in_data(:,:) = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

    ! Create a tensor based off an input array
    call torch_tensor_from_array(tensor, in_data, device_type)

    ! Call the torch_tensor_zero subroutine using its class method alias
    call tensor%zero()

    ! Compare the data in the tensor to the input array
    expected(:,:) = 0.0
    call check(error, assert_allclose(in_data, expected, test_name="test_torch_tensor_zero"))

  end subroutine test_torch_tensor_zero

end module unittest_tensor_manipulation
