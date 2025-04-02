program tensor_manipulation

  ! Import the FTorch procedures that are used in this worked example
  use ftorch, only: assignment(=), operator(+), torch_kCPU, torch_kFloat32, torch_tensor, &
                    torch_tensor_delete, torch_tensor_empty, torch_tensor_from_array, &
                    torch_tensor_ones, torch_tensor_print

  use, intrinsic :: iso_c_binding, only: c_int64_t

  ! Import the real32 type for 32-bit floating point numbers
  use, intrinsic :: iso_fortran_env, only: sp => real32

  implicit none

  ! Set working precision for reals to be 32-bit floats
  integer, parameter :: wp = sp

  ! Define some tensors
  type(torch_tensor) :: a, b, c

  ! Variables for constructing tensors with torch_tensor_ones
  integer, parameter :: ndims = 2
  integer(c_int64_t), dimension(2), parameter :: tensor_shape = [2, 3]

  ! Variables for constructing tensors with torch_tensor_from_array
  real(wp), dimension(2,3), target :: in_data, out_data

  ! Create a tensor of ones
  ! -----------------------
  ! Doing the same for a tensor of zeros is as simple as adding the torch_tensor_zeros subroutine
  ! to the list of imports and switching out the following subroutine call.
  call torch_tensor_ones(a, ndims, tensor_shape, torch_kFloat32, torch_kCPU)

  ! Print the contents of the tensor
  ! --------------------------------
  ! This will show the tensor data as well as its device type, data type, and shape.
  write(*,*) "Contents of first input tensor:"
  call torch_tensor_print(a)

  ! Create a tensor based off an array
  ! ----------------------------------
  ! Note that the API is slightly different for this subroutine. In particular, the dimension,
  ! shape and data type of the tensor are automatically inherited from the input array so do not
  ! need to be specified.
  in_data(:,:) = reshape([1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp, 6.0_wp], [2,3])
  call torch_tensor_from_array(b, in_data, torch_kCPU)
  ! Another way of viewing the contents of a tensor is to print the array used as its input.
  write(*,*) "Contents of second input tensor:"
  write(*,*) in_data

  ! Extract data from the tensor as a Fortran array
  ! -----------------------------------------------
  ! This requires some setup in advance. Create a tensor based off the Fortran array that you want
  ! to extract data into in the same way as above. There's no need to assign values to the array.
  call torch_tensor_from_array(c, out_data, torch_kCPU)

  ! Perform arithmetic on the tensors using the overloaded addition operator
  ! ------------------------------------------------------------------------
  ! Note that if the output tensor hasn't been constructed as above then it will be automatically
  ! constructed using `torch_tensor_empty` but it won't be possible to extract its data into an
  ! array.
  c = a + b
  write(*,*) "Output:"
  write(*,*) out_data

  ! Clean up
  ! --------
  ! It's good practice to free the memory associated with the tensors after use. However, with
  ! recent versions of FTorch calling `torch_tensor_delete` is optional because it has been set up
  ! to be called automatically when the tensor goes out of scope.
  call torch_tensor_delete(a)
  call torch_tensor_delete(b)
  call torch_tensor_delete(c)

  write(*,*) "Tensor manipulation example ran successfully"

end program tensor_manipulation
