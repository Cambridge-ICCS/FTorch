!| Unit tests for FTorch's non-overloaded operators acting on tensors.
!
!  * License
!    FTorch is released under an MIT license.
!    See the [LICENSE](https://github.com/Cambridge-ICCS/FTorch/blob/main/LICENSE)
!    file for details.
module unittest_tensor_operators
  use ftorch, only: assignment(=), torch_kCPU, torch_kFloat32, torch_tensor, torch_tensor_from_array
  use ftorch_test_utils, only: assert_allclose

  use, intrinsic :: iso_fortran_env, only: sp => real32
  use, intrinsic :: iso_c_binding, only : c_int64_t

  implicit none

  private
  public :: collect_tensor_operators_suite

  ! Set working precision for reals
  integer, parameter :: wp = sp

  ! All unit tests in this module run on CPU
  integer, parameter :: device_type = torch_kCPU

  ! All unit tests in this module use 2D arrays with float32 precision
  integer, parameter :: ndims = 2
  integer, parameter :: dtype = torch_kFloat32

contains

  !> The test suite for the mesh_generator
  function test_tensor_operators() result(tests)
    use veggies, only: describe, it, test_item_t, example_t
    implicit none
    type(test_item_t) :: tests

    tests = describe( &
            "test_tensor_operators", &
            [ it( &
                "test_torch_tensor_sum passes", &
                test_torch_tensor_sum) &
            , it( &
                "test_torch_tensor_mean passes", &
                test_torch_tensor_mean) &
            ])
  end function test_tensor_operators

  function test_torch_tensor_sum() result(result_)
    use ftorch, only: torch_tensor_sum
    use veggies, only: result_t, assert_that

    type(result_t) :: result_
    type(torch_tensor) :: in_tensor, out_tensor
    real(wp), dimension(2,3), target :: in_data
    real(wp), dimension(1), target :: out_data
    real(wp), dimension(2,3) :: expected2d
    real(wp), dimension(1) :: expected1d
    logical :: test_pass

    ! Create two arbitrary input arrays
    in_data(:,:) = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

    ! Create tensors based off the input array
    call torch_tensor_from_array(in_tensor, in_data, device_type)

    ! Create a tensor based off an output array containing a single scalar value
    call torch_tensor_from_array(out_tensor, out_data, device_type)

    ! Compute the sum over the entries in the first tensor and assign it to the single value in the
    ! second
    call torch_tensor_sum(out_tensor, in_tensor)

    ! Check input arrays are unchanged by the summation
    expected2d(:,:) = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    result_ = assert_that(assert_allclose(in_data, expected2d, test_name="test_torch_tensor_sum"))
    if (.not. result_%passed) then
      return
    end if

    ! Compare the data in the tensor to the expected sum
    expected1d(:) = 21.0
    result_ = assert_that(assert_allclose(out_data, expected1d, test_name="test_torch_tensor_sum"))

  end function test_torch_tensor_sum

  subroutine test_torch_tensor_mean(error)
    use ftorch, only: torch_tensor_mean
    use veggies, only: result_t, assert_that

    type(result_t) :: result_
    type(torch_tensor) :: in_tensor, out_tensor
    real(wp), dimension(2,3), target :: in_data
    real(wp), dimension(1), target :: out_data
    real(wp), dimension(2,3) :: expected2d
    real(wp), dimension(1) :: expected1d
    logical :: test_pass

    ! Create two arbitrary input arrays
    in_data(:,:) = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

    ! Create tensors based off the input array
    call torch_tensor_from_array(in_tensor, in_data, device_type)

    ! Create a tensor based off an output array containing a single scalar value
    call torch_tensor_from_array(out_tensor, out_data, device_type)

    ! Compute the mean over the entries in the first tensor and assign it to the single value in the
    ! second
    call torch_tensor_mean(out_tensor, in_tensor)

    ! Check input arrays are unchanged by the mean operation
    expected2d(:,:) = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    result_ = assert_that(assert_allclose(in_data, expected2d, test_name="test_torch_tensor_mean"))
    if (.not. result_%passed) then
      return
    end if

    ! Compare the data in the tensor to the expected mean
    expected1d(:) = 21.0 / 6.0
    result_ = assert_that(assert_allclose(out_data, expected1d, test_name="test_torch_tensor_mean"))

  end subroutine test_torch_tensor_mean

end module unittest_tensor_operators
