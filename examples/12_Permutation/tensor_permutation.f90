program tensor_permutation

  ! Import the FTorch procedures that are used in this worked example
  use ftorch, only: assignment(=), torch_kCPU, torch_kFloat32, &
                    torch_tensor, torch_tensor_delete, torch_tensor_from_array, &
                    torch_tensor_print

  ! Import the real32 type for 32-bit floating point numbers
  use, intrinsic :: iso_fortran_env, only: sp => real32

  implicit none

  ! Set working precision for reals to be 32-bit floats
  integer, parameter :: wp = sp

  ! Define some tensors
  type(torch_tensor) :: a, b, c, d, e

  ! Variables for constructing tensors with torch_tensor_from_array
  real(wp), dimension(2,3), target :: in_data, out_data

  ! Variables for constructing permuted tensors with torch_tensor_from_array
  real(wp), dimension(3,2), target :: out_data_permuted

  ! Loop index for array printing
  integer :: i

  ! We start by constructing a 2D Fortran array in_data of shape [2, 3]:
  in_data(:,:) = reshape([1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp, 6.0_wp], [2,3])

  write(*,*) "2D Fortran array in_data:"
  do i = 1, 2
    write(*,fmt='(3F6.1)') in_data(i, :)
  end do

  call torch_tensor_from_array(a, in_data, torch_kCPU)

  write(*,*) "Corresponding Torch tensor a constructed from in_data:"
  call torch_tensor_print(a)

  write(*,*) "Shape of tensor, a, from in_data:", a%get_shape()   ! Expected: 2 3
  write(*,*) "Stride of tensor, a, from in_data:", a%get_stride() ! Expected: 1 2 (column-major)
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
  call torch_tensor_from_array(b, in_data, torch_kCPU, permute_dims=[2, 1])
  write(*,*) "Torch tensor b constructed from in_data with permutation [2, 1]:"
  call torch_tensor_print(b)
  write(*,*) "Shape of permuted tensor, b:", b%get_shape()   ! Expected: 3 2
  write(*,*) "Stride of permuted tensor, b:", b%get_stride() ! Expected: 2 1 (row-major)
  write(*,*)

  ! Note that the permute_dims argument is validated: it must be a permutation of
  ! [1, ..., rank]. Uncomment either of the following lines to trigger an error:
  ! call torch_tensor_from_array(b, in_data, torch_kCPU, permute_dims=[3, 1])  ! out of range
  ! call torch_tensor_from_array(b, in_data, torch_kCPU, permute_dims=[1, 1])  ! duplicate

  ! To extract the permuted data back into a Fortran array, the output array
  ! MUST be declared with the PERMUTED shape. Here out_data_permuted has shape
  ! [3, 2] to match the tensor. Using the original shape [2, 3] would cause a
  ! memory layout mismatch.
  call torch_tensor_from_array(c, out_data_permuted, torch_kCPU)
  c = b
  write(*,*) "Permuted tensor data extracted to Fortran array, out_data_permuted:"
  do i = 1, 3
    write(*,fmt='(2F6.1)') out_data_permuted(i, :)
  end do
  write(*,*)

  ! Note when we say MUST be declared with the PERMUTED shape that is not strictly true.
  ! We could have declared the tensor we copy back to to also be permuted from its host
  ! array in the same way as the following shows:
  call torch_tensor_from_array(d, out_data, torch_kCPU, permute_dims=[2, 1])
  d = b
  write(*,*) "Permuted tensor data extracted to Fortran array, out_data, via a permutation:"
  do i = 1, 2
    write(*,fmt='(3F6.1)') out_data(i, :)
  end do
  ! We recover our original [2, 3] Fortran array in_data via permuting both into and out of Torch!"
  write(*,*)

  ! Note that when using the assignment operator the LHS tensor will NOT copy over the
  ! strides from the RHS and will be constructed with C-style row-major ordering.
  ! The shape and values of each element are copied.
  e = a  ! Create Torch tensor e not associated to a Fortran array
  write(*,*) "Shape of FTorch-constructed tensor, a:", a%get_shape()   ! Expected: 2 3
  write(*,*) "Stride of FTorch-constructed tensor, a:", a%get_stride() ! Expected: 1 2
  write(*,*) "Contents of Torch tensor a:"
  call torch_tensor_print(a)
  write(*,*) "Shape of copied Torch tensor, e:", e%get_shape()   ! Expected: 2 3
  write(*,*) "Stride of copied Torch tensor, e:", e%get_stride() ! Expected: 3 1
  write(*,*) "Contents of Torch tensor e:"
  call torch_tensor_print(e)
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
  write(*,*) "Torch tensor a appearing identical to the Fortran array:"
  call a%print()
  write(*,*) "Stride of tensor a:", a%get_stride()
  write(*,*)

  ! 4. If we wanted, instead, for the Torch tensor to be constructed with native
  !    row-major strides through the underlying memory then we need to apply a
  !    permutation through the permute_dims argument. This will give us, in Torch,
  !    the transpose of the array we had in Fortran, as can be seen here:
  write(*,*) "Permuted Torch tensor b (involution, appearing to Torch as a transpose of the Fortran array):"
  call b%print()
  write(*,*) "Stride of tensor b:", b%get_stride()
  write(*,*)

  ! Clean up
  ! --------
  call torch_tensor_delete(a)
  call torch_tensor_delete(b)
  call torch_tensor_delete(c)
  call torch_tensor_delete(d)
  call torch_tensor_delete(e)

  write(*,*) "Tensor permutation example ran successfully"

end program tensor_permutation
