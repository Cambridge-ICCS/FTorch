program example
  use ftorch

  real, dimension(2), target :: out_data1
  type(torch_tensor) :: a, b, Q, multiplier, divisor

  ! Initialise Torch Tensors from input arrays as in Python example
  call torch_tensor_from_array(a, [2.0, 3.0], [1], torch_kCPU, requires_grad=.true.)
  call torch_tensor_from_array(b, [6.0, 4.0], [1], torch_kCPU, requires_grad=.true.)

  ! Initialise Torch Tensor from array used for output
  call torch_tensor_from_array(Q, out_data1, [1], torch_kCPU)

  ! Scalar multiplication and division are not currently implemented in FTorch. However, you can
  ! achieve the same thing by defining a rank-1 tensor with a single entry, as follows:
  call torch_tensor_from_array(multiplier, [3.0], [1], torch_kCPU)
  call torch_tensor_from_array(divisor, [3.0], [1], torch_kCPU)

  ! Compute the same mathematical expression as in the Python example
  Q = multiplier * (a**3 - b * b / divisor)

  ! Run the back-propagation operator
  call torch_tensor_backward(Q)
  call torch_tensor_print(a%grad())
  call torch_tensor_print(b%grad())

end program example
