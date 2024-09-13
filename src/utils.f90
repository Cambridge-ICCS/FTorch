!| Utils module for FTorch containing assertions for testing
!
!  * License  
!    FTorch is released under an MIT license.
!    See the [LICENSE](https://github.com/Cambridge-ICCS/FTorch/blob/main/LICENSE)
!    file for details.

module utils

  implicit none

  contains

    !> Print the result of a test to the terminal
    subroutine test_print(test_name, relative_error, test_pass)

      character(len=*), intent(in) :: test_name !! Name of the test being run
      real, intent(in) :: relative_error !! Relative error used for assertion
      logical, intent(in) :: test_pass !! Result of the assertion

      character(len=15) :: report

      if (test_pass) then
        report = char(27)//'[32m'//'PASSED'//char(27)//'[0m'
      else
        report = char(27)//'[31m'//'FAILED'//char(27)//'[0m'
      end if
      write(*, '(A, " :: [", A, "] maximum relative error = ", E11.4)') report, trim(test_name), relative_error
    end subroutine test_print

   !> Asserts that two real values coincide to a given relative tolerance
   function assert_real(got, expect, test_name, rtol, print_result) result(test_pass)

      character(len=*), intent(in) :: test_name !! Name of the test being run
      real, intent(in) :: got !! The value to be tested
      real, intent(in) :: expect !! The expected value
      real, optional :: rtol !! Optional relative tolerance (defaults to 1e-5)
      logical, optional :: print_result !! Optionally print test result to screen (defaults to .true.)

      logical :: test_pass !! Did the assertion pass

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

      test_pass = (abs(got - expect) <= rtol_value * expect)

      if (print_result_value) then
        call test_print(test_name, rtol_value, test_pass)
      end if

   end function assert_real

end module utils
