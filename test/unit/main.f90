program unit_test_suite
  use, intrinsic :: iso_fortran_env, only : error_unit
  use testdrive, only : run_testsuite, new_testsuite, testsuite_type
  use unittest_tensor_operators, only : collect_tensor_operators_suite
  use unittest_tensor_manipulation, only : collect_tensor_manipulation_suite
  implicit none
  integer :: stat, is
  type(testsuite_type), allocatable :: testsuites(:)
  character(len=*), parameter :: fmt = '("#", *(1x, a))'

  stat = 0

  testsuites = [ &
    ! TODO: "tensor_constructors_destructors"
    ! TODO: "tensor_interrogation"
    new_testsuite("test_tensor_operators", collect_tensor_operators_suite), &
    new_testsuite("test_tensor_manipulation", collect_tensor_manipulation_suite) &
    ! TODO: "tensor_operator_overloads"
    ! TODO: "tensor_operator_overloads_autograd"
    ! TODO: "tensor_operators_autograd"
  ]

  do is = 1, size(testsuites)
    write(error_unit, fmt) "Testing:", testsuites(is)%name
    call run_testsuite(testsuites(is)%collect, error_unit, stat)
  end do

  if (stat > 0) then
    write(error_unit, '(i0, 1x, a)') stat, "test(s) failed!"
    error stop
  end if

end program unit_test_suite
