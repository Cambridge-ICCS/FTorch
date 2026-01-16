program unit_test_suite
    use veggies, only : test_item_t, test_that, run_tests
    use unittest_tensor_operators, only: test_tensor_operators
    ! TODO: use unittest_tensor_constructors_destructors, only: test_tensor_constructors_destructors
    ! TODO: use unittest_tensor_interrogation, only: test_tensor_interrogation
    ! TODO: use unittest_tensor_manipulation, only: test_tensor_manipulation
    ! TODO: use unittest_tensor_operator_overloads, only: test_tensor_operator_overloads
    ! TODO: use unittest_tensor_operator_overloads_autograd, only: test_tensor_operator_overloads_autograd
    ! TODO: use unittest_tensor_operators_autograd, only: test_tensor_operators_autograd
    implicit none

    if (.not. run()) then
      stop 1
    end if

contains
    function run() result(passed)
        logical :: passed

        type(test_item_t) :: tests
        type(test_item_t) :: individual_tests(1)

        individual_tests(1) = test_tensor_operators()

        tests = test_that(individual_tests)

        passed = run_tests(tests)
    end function run
end program unit_test_suite
