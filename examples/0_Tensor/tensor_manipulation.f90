program tensor_manipulation

  ! Import the FTorch procedures that are used in this worked example
  use ftorch, only: assignment(=), operator(+), torch_kCPU, torch_kFloat32, torch_tensor, &
                    torch_tensor_delete, torch_tensor_empty, torch_tensor_from_array, &
                    torch_tensor_ones, torch_tensor_print, torch_tensor_to_array

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
  integer, parameter :: tensor_layout(ndims) = [1, 2]
  real(wp), dimension(2,3), target :: in_data

  ! Array for extracting an array from a tensor
  real(wp), dimension(:,:), pointer :: out_data

  ! Create a tensor of ones
  ! -----------------------
  ! Doing the same for a tensor of zeros is as simple as adding the torch_tensor_zeros subroutine
  ! to the list of imports and switching out the following subroutine call.
  call torch_tensor_ones(a, ndims, tensor_shape, torch_kFloat32, torch_kCPU)

  ! Print the contents of the tensor
  ! --------------------------------
  write(*,*) "Contents of first input tensor:"
  call torch_tensor_print(a)

  ! Create a tensor based off an array
  ! ----------------------------------
  ! Note that the API is slightly different for this subroutine. In particular, the dimension,
  ! shape and data type of the tensor are automatically inherited from the input array. Further,
  ! the tensor layout should be specified, which determines the indexing order.
  in_data(:,:) = reshape([1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp, 6.0_wp], [2,3])
  call torch_tensor_from_array(b, in_data, tensor_layout, torch_kCPU)
  ! Another way of viewing the contents of a tensor is to print the array used as its input.
  write(*,*) "Contents of second input tensor:"
  write(*,*) in_data

  ! Perform arithmetic on the tensors using the overloaded addition operator
  ! ------------------------------------------------------------------------
  ! It's important that the tensor used for the sum has been constructed. It's sufficient to use
  ! torch_tensor_empty, which leaves its values unset. Note that it's required to import the
  ! overloaded assignment and addition operators for this to work correctly.
  call torch_tensor_empty(c, ndims, tensor_shape, torch_kFloat32, torch_kCPU)
  c = a + b

  ! Extract data from the tensor as a Fortran array
  ! -----------------------------------------------
  ! Note that the torch_tensor_to_array subroutine will allocate the output array to the
  ! appropriate size if it hasn't already been allocated.
  call torch_tensor_to_array(c, out_data, shape(in_data), tensor_layout)
  write(*,*) "Output:"
  write(*,*) out_data

  ! Clean up
  ! --------
  ! It's good practice to free the memory associated with the tensors after use. We should also
  ! nullify any pointers, such as those required by torch_tensor_to_array.
  call torch_tensor_delete(a)
  call torch_tensor_delete(b)
  call torch_tensor_delete(c)
  nullify(out_data)

  write(*,*) "Tensor manipulation example ran successfully"

end program tensor_manipulation
