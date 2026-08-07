program tensor_manipulation

  ! Import the FTorch procedures that are used in this worked example
  use ftorch, only: assignment(=), operator(+), ftorch_int, torch_kCPU, torch_kFloat32, &
                    torch_tensor, torch_tensor_delete, torch_tensor_empty, &
                    torch_tensor_from_array, torch_tensor_mean, torch_tensor_ones, &
                    torch_tensor_print

  use, intrinsic :: iso_c_binding, only: c_int64_t

  ! Import the real32 type for 32-bit floating point numbers
  use, intrinsic :: iso_fortran_env, only: sp => real32

  implicit none

  ! Set working precision for reals to be 32-bit floats
  integer, parameter :: wp = sp

  ! Define some tensors
  type(torch_tensor) :: a, b, c, d, e, f, g, mean

  ! Variables for constructing tensors with torch_tensor_ones
  integer, parameter :: ndims = 2
  integer(c_int64_t), dimension(2), parameter :: tensor_shape = [2, 3]

  ! Variables for constructing tensors with torch_tensor_from_array
  real(wp), dimension(2,3), target :: in_data, out_data

  ! Variables for constructing permuted tensors with torch_tensor_from_array
  real(wp), dimension(3,2), target :: out_data_permuted

  ! Variables for constructing tensors containing a single scalar value
  real(wp), dimension(1), target :: scalar_data

  ! Loop index for array printing
  integer :: i

  ! Create a tensor of ones
  ! -----------------------
  ! Doing the same for a tensor of zeros is as simple as adding the torch_tensor_zeros subroutine
  ! to the list of imports and switching out the following subroutine call.
  call torch_tensor_ones(a, ndims, tensor_shape, torch_kFloat32, torch_kCPU)

  ! Note that the tensor had memory allocated on the Torch side hence it is represented
  ! in row-major order.
  write(*,*) "Shape of the ones tensor:", a%get_shape()
  write(*,*) "Stride of the ones tensor:", a%get_stride()  ! Expected 3, 1 (row-major)

  ! Print the contents of the tensor
  ! --------------------------------
  ! This will show the tensor data as well as its device type, data type, and shape.
  write(*,*) "Contents of the ones tensor:"
  call torch_tensor_print(a)
  write(*,*)

  ! Create a tensor based off an array
  ! ----------------------------------
  ! Note that the API is slightly different for this subroutine. In particular, the dimension,
  ! shape and data type of the tensor are automatically inherited from the input array so do not
  ! need to be specified.
  in_data(:,:) = reshape([1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp, 6.0_wp], [2,3])
  call torch_tensor_from_array(b, in_data, torch_kCPU)
  ! The torch_tensor_print subroutine can also be called as a module procedure
  write(*,*) "Contents of second input tensor, b:"
  call b%print()

  ! For a tensor build on top of a Fortran array, the underlying data is in column-major order
  ! Since FTorch performs no copies, the strides of the tensor will also correspond to column-major
  ! order. This is different from the default behaviour of Torch which builds tensors with row-major
  ! order by default.
  write(*,*) "Shape of the second tensor from Fortran array:", b%get_shape()   ! Expected: 2 3
  write(*,*) "Stride of the second tensor from Fortran array:", b%get_stride() ! Expected: 1 2
  write(*,*)

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
  !
  ! Another way of viewing the contents of a tensor is to print the array associated with it.
  c = a + b
  write(*,*) "Sum of input tensors:"
  do i = 1, 2
    write(*,fmt='(3F6.1)') out_data(i, :)
  end do
  write(*,*)

  ! Taking the mean of the contents of a tensor
  ! ---------------------------------
  ! In addition to the overloaded mathematical operators, FTorch provides other operators for
  ! computations involving tensors. One such operator is the mean over the values in the tensor.
  ! While the mean over a tensor is a scalar value, we compute it as a single value in a tensor of
  ! dimension 1. First construct the 'tensor' with `torch_tensor_from_array` and then compute the
  ! mean as follows.
  call torch_tensor_from_array(mean, scalar_data, torch_kCPU)
  call torch_tensor_mean(mean, c)
  write(*,*) "Mean value:"
  write(*,fmt='(1F6.1)') scalar_data
  write(*,*)

  ! Clean up
  ! --------
  ! It's good practice to free the memory associated with the tensors after use. However, with
  ! recent versions of FTorch calling `torch_tensor_delete` is optional because it has been set up
  ! to be called automatically when the tensor goes out of scope.
  call torch_tensor_delete(a)
  call torch_tensor_delete(b)
  call torch_tensor_delete(c)


  ! ===================================================================================
  ! For new users who want to perform basic tensor and model operations using FTorch,
  ! it is sufficcient to end the example here.
  ! The following section details advanced use regarding memory layouts and
  ! optimisation considerations and should only be used by those who know what they are
  ! doing and are familiar with the library.
  ! ===================================================================================

  ! We start by reconstructing tensor b from the above exercises, as a Torch tensor
  ! representation of the 2D in_data Fortran array:
  write(*,*) "2D Fortran array in_data:"
  do i = 1, 2
    write(*,fmt='(3F6.1)') in_data(i, :)
  end do

  call torch_tensor_from_array(b, in_data, torch_kCPU)

  write(*,*) "Corresponding Torch tensor b constructed from in_data:"
  call torch_tensor_print(b)

  write(*,*) "Shape of tensor, b, from in_data:", b%get_shape()   ! Expected: 2 3
  write(*,*) "Stride of tensor, b, from in_data:", b%get_stride() ! Expected: 1 2 (column-major)
  write(*,*)

  ! Permuting tensors
  ! -----------------
  ! torch_tensor_from_array accepts an optional permute_dims argument that permutes
  ! the dimensions of the input Fortran array to change how the resulting tensor
  ! appears in Torch. This is useful when the layout of the Fortran array does not
  ! match the layout expected by a Torch model, or if manipulation is desired for
  ! optimal indexing/striding. It operates in a similar way to torch.permute(), but
  ! indexing from 1 instead of 0!
  !
  ! Here we create a tensor from in_data (shape [2, 3]) with permute_dims=[2, 1],
  ! which is an involution (i.e. a transpose). The resulting tensor has shape
  ! [3, 2]. FTorch achieves this without changing memory by setting the strides for
  ! the Torch tensor to be row-major in nature:
  call torch_tensor_from_array(d, in_data, torch_kCPU, permute_dims=[2, 1])
  write(*,*) "Torch tensor d constructed from in_data with permutation [2, 1]:"
  call torch_tensor_print(d)
  write(*,*) "Shape of permuted tensor, d:", d%get_shape()   ! Expected: 3 2
  write(*,*) "Stride of permuted tensor, d:", d%get_stride() ! Expected: 2 1 (row-major)
  write(*,*)

  ! Note that the permute_dims argument is validated: it must be a permutation of
  ! [1, ..., rank]. Uncomment either of the following lines to trigger an error:
  ! call torch_tensor_from_array(d, in_data, torch_kCPU, permute_dims=[3, 1])  ! out of range
  ! call torch_tensor_from_array(d, in_data, torch_kCPU, permute_dims=[1, 1])  ! duplicate

  ! To extract the permuted data back into a Fortran array, the output array
  ! MUST be declared with the PERMUTED shape. Here out_data_permuted has shape
  ! [3, 2] to match the tensor. Using the original shape [2, 3] would cause a
  ! memory layout mismatch.
  call torch_tensor_from_array(e, out_data_permuted, torch_kCPU)
  e = d
  write(*,*) "Permuted tensor data extracted to Fortran array, out_data_permuted:"
  do i = 1, 3
    write(*,fmt='(2F6.1)') out_data_permuted(i, :)
  end do
  write(*,*)

  ! Note when we say MUST be declared with the PERMUTED shape that is not strictly true.
  ! We could have declared the tensor we copy back to to also be permuted from its host
  ! array in the same way as the following shows:
  call torch_tensor_from_array(f, out_data, torch_kCPU, permute_dims=[2, 1])
  f = d
  write(*,*) "Permuted tensor data extracted to Fortran array, out_data, via a permutation:"
  do i = 1, 2
    write(*,fmt='(3F6.1)') out_data(i, :)
  end do
  ! We recover our original [2, 3] Fortran array in_data via permuting both into and out of Torch!"
  write(*,*)

  ! Note that when using the assignment operator the LHS tensor will NOT copy over the
  ! strides from the RHS and will be constructed with C-style row-major ordering.
  ! The shape and values of each element are copied.
  g = b  ! Create Torch tensor g not associated to a Fortran array
  write(*,*) "Shape of FTorch-constructed tensor, b:", b%get_shape()   ! Expected: 2 3
  write(*,*) "Stride of FTorch-constructed tensor, b:", b%get_stride() ! Expected: 1 2
  write(*,*) "Contents of Torch tensor b:"
  call torch_tensor_print(b)
  write(*,*) "Shape of copied Torch tensor, g:", g%get_shape()   ! Expected: 2 3
  write(*,*) "Stride of copied Torch tensor, g:", g%get_stride() ! Expected: 3 1
  write(*,*) "Contents of Torch tensor g:"
  call torch_tensor_print(g)
  write(*,*)

  ! Fortran vs. Torch memory layout
  ! -------------------------------
  ! Fortran stores arrays in column-major order: elements within the same
  ! column are contiguous in memory. Torch stores tensors in row-major order:
  ! elements within the same row are contiguous in memory. When FTorch wraps a
  ! Fortran array as a Torch tensor without copying, both sides refer to the
  ! same data in memory — but can interpret it differently.
  !
  ! To try and make this clear we print the data three ways:
  !   1. Row by row: the logical Fortran array in_data shape [2, 3].
  !   2. As it appears in memory: 6 floats in column-major order.
  !   3. As it appears in Torch unpermuted: achieved by taking column-major strides
  !   4. As it appears in Torch permuted: achieved by taking row-major strides

  ! 1. Logical matrix view: print one row per line.
  !    This is how we think about the data conceptually.
  write(*,*) "Fortran [2, 3] array in_data, printed row by row:"
  do i = 1, 2
    write(*,fmt='(3F6.1)') in_data(i, :)
  end do
  write(*,*)

  ! 2. Memory-order view.
  !    Fortran stores column 1 first, then column 2, then column 3.
  !    This is the exact byte sequence Torch receives.
  write(*,*) "Fortran array in_data, as it appears in memory:"
  write(*,fmt='(6F6.1)') in_data
  write(*,*)

  ! 3. A Torch tensor created using torch_tensor_from_array with no permute_dims
  !    will have the same shape and elements as the Fortran array.
  !    Both share the same underlying memory, and this is achieved by FTorch setting
  !    the strides of the Torch tensor to be column major (see below).
  !    For details on implementation see the source code and torch_tensor_from_blob.
  write(*,*) "Torch tensor b appearing identical to the Fortran array:"
  call b%print()
  write(*,*) "Stride of tensor b:", b%get_stride()
  write(*,*)

  ! 4. If we wanted, instead, for the Torch tensor to be constructed with native
  !    row-major strides through the underlying memory then we need to apply a
  !    permutation through the permute_dims argument. This will give us, in Torch,
  !    the transpose of the array we had in Fortran, as can be seen here:
  write(*,*) "Permuted Torch tensor d (involution, appearing to Torch as a transpose of the Fortran array):"
  call d%print()
  write(*,*) "Stride of tensor d:", d%get_stride()
  write(*,*)

  ! Clean up
  ! --------
  call torch_tensor_delete(b)
  call torch_tensor_delete(d)
  call torch_tensor_delete(e)
  call torch_tensor_delete(f)
  call torch_tensor_delete(g)

  write(*,*) "Tensor manipulation example ran successfully"

end program tensor_manipulation
