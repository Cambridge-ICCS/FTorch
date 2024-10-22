!| Utils module for FTorch containing assertions for testing
!
!  * License  
!    FTorch is released under an MIT license.
!    See the [LICENSE](https://github.com/Cambridge-ICCS/FTorch/blob/main/LICENSE)
!    file for details.

module ftorch_test_utils

  implicit none

  interface assert_real_array
    module procedure assert_real_array_1d
    module procedure assert_real_array_2d
  end interface

  contains

    !> Print the result of a test to the terminal
    subroutine test_print(test_name, message, test_pass)

      character(len=*), intent(in) :: test_name !! Name of the test being run
      character(len=*), intent(in) :: message !! Message to print
      logical, intent(in) :: test_pass !! Result of the assertion

      character(len=15) :: report

      if (test_pass) then
        report = char(27)//'[32m'//'PASSED'//char(27)//'[0m'
      else
        report = char(27)//'[31m'//'FAILED'//char(27)//'[0m'
      end if
      write(*, '(A, " :: [", A, "] ", A)') report, trim(test_name), trim(message)
    end subroutine test_print

    !> Asserts that two real values coincide to a given relative tolerance
    function assert_real(got, expect, test_name, rtol, print_result) result(test_pass)

      character(len=*), intent(in) :: test_name !! Name of the test being run
      real, intent(in) :: got !! The value to be tested
      real, intent(in) :: expect !! The expected value
      real, intent(in), optional :: rtol !! Optional relative tolerance (defaults to 1e-5)
      logical, intent(in), optional :: print_result !! Optionally print test result to screen (defaults to .true.)

      logical :: test_pass !! Did the assertion pass?

      character(len=80) :: message

      real :: relative_error
      real :: rtol_value
      logical :: print_result_value

      if (.not. present(rtol)) then
        rtol_value = 1e-5
      else
        rtol_value = rtol
      end if

      if (.not. present(print_result)) then
        print_result_value = .true.
      else
        print_result_value = print_result
      end if

      test_pass = (abs(got - expect) <= rtol_value * abs(expect))

      if (print_result_value) then
        write(message,'("relative tolerance = ", E11.4)') rtol_value
        call test_print(test_name, message, test_pass)
      end if

    end function assert_real

    !> Asserts that two real-valued 1D arrays coincide to a given relative tolerance
    function assert_real_array_1d(got, expect, test_name, rtol, print_result) result(test_pass)

      character(len=*), intent(in) :: test_name !! Name of the test being run
      real, intent(in), dimension(:) :: got !! The array of values to be tested
      real, intent(in), dimension(:) :: expect !! The array of expected values
      real, intent(in), optional :: rtol !! Optional relative tolerance (defaults to 1e-5)
      logical, intent(in), optional :: print_result !! Optionally print test result to screen (defaults to .true.)

      logical :: test_pass !! Did the assertion pass?

      character(len=80) :: message

      real :: relative_error
      real :: rtol_value
      integer :: shape_error
      logical :: print_result_value

      if (.not. present(rtol)) then
        rtol_value = 1.0e-5
      else
        rtol_value = rtol
      end if

      if (.not. present(print_result)) then
        print_result_value = .true.
      else
        print_result_value = print_result
      end if

      ! Check the shapes of the arrays match
      shape_error = maxval(abs(shape(got) - shape(expect)))
      test_pass = (shape_error == 0)

      if (test_pass) then
        test_pass = all(abs(got - expect) <= rtol_value * abs(expect))
        if (print_result_value) then
          write(message,'("relative tolerance = ", E11.4)') rtol_value
          call test_print(test_name, message, test_pass)
        end if
      else if (print_result_value) then
        call test_print(test_name, "Arrays have mismatching shapes.", test_pass)
      endif

    end function assert_real_array_1d

    !> Asserts that two real-valued 2D arrays coincide to a given relative tolerance
    function assert_real_array_2d(got, expect, test_name, rtol, print_result) result(test_pass)

      character(len=*), intent(in) :: test_name !! Name of the test being run
      real, intent(in), dimension(:,:) :: got !! The array of values to be tested
      real, intent(in), dimension(:,:) :: expect !! The array of expected values
      real, intent(in), optional :: rtol !! Optional relative tolerance (defaults to 1e-5)
      logical, intent(in), optional :: print_result !! Optionally print test result to screen (defaults to .true.)

      logical :: test_pass !! Did the assertion pass?

      character(len=80) :: message

      real :: relative_error
      real :: rtol_value
      integer :: shape_error
      logical :: print_result_value

      if (.not. present(rtol)) then
        rtol_value = 1.0e-5
      else
        rtol_value = rtol
      end if

      if (.not. present(print_result)) then
        print_result_value = .true.
      else
        print_result_value = print_result
      end if

      ! Check the shapes of the arrays match
      shape_error = maxval(abs(shape(got) - shape(expect)))
      test_pass = (shape_error == 0)

      if (test_pass) then
        test_pass = all(abs(got - expect) <= rtol_value * abs(expect))
        if (print_result_value) then
          write(message,'("relative tolerance = ", E11.4)') rtol_value
          call test_print(test_name, message, test_pass)
        end if
      else if (print_result_value) then
        call test_print(test_name, "Arrays have mismatching shapes.", test_pass)
      endif

    end function assert_real_array_2d


end module ftorch_test_utils
