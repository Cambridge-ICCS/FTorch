program example
  use ftorch
  implicit none

  integer :: tensor_layout(1) = [1]
  type(torch_tensor) :: a, b, Q

  call torch_tensor_from_array(a, [2.0, 3.0], tensor_layout, torch_kCPU,       &
                               device_index=0) ! FIXME: requires_grad=.true.
  call torch_tensor_from_array(b, [6.0, 4.0], tensor_layout, torch_kCPU,       &
                               device_index=0) ! FIXME: requires_grad=.true.

  ! TODO: Q = 3 * a ** 3 - 2 * b
  !   Requires overloading elementary operations
  print *, "a ="
  call torch_tensor_print(a)
  print *, "b ="
  call torch_tensor_print(b)
  ! ---
  Q = a
  print *, "Q = a ="
  call torch_tensor_print(Q)
  print *, "a ="
  call torch_tensor_print(a)
  print *, "b ="
  call torch_tensor_print(b)
  ! ---
  print *, "Q = a + b ="
  Q = a + b
  call torch_tensor_print(Q)
  print *, "a ="
  call torch_tensor_print(a)
  print *, "b ="
  call torch_tensor_print(b)
  ! ---
  print *, "Q = a * b ="
  Q = a * b
  call torch_tensor_print(Q)
  print *, "a ="
  call torch_tensor_print(a)
  print *, "b ="
  call torch_tensor_print(b)
  ! ---
  print *, "Q = a - b ="
  Q = a - b
  call torch_tensor_print(Q)
  print *, "a ="
  call torch_tensor_print(a)
  print *, "b ="
  call torch_tensor_print(b)
  ! ---
  print *, "Q = a / b ="
  Q = a / b
  call torch_tensor_print(Q)
  print *, "a ="
  call torch_tensor_print(a)
  print *, "b ="
  call torch_tensor_print(b)

  ! TODO: Backward
  !   Requires API extension

end program example
