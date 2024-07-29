module test_ftorch
  use pFUnit
  use ftorch

  implicit none

  @test
  subroutine test_torch_tensor_zeros()
    type(torch_tensor) :: tensor
    integer(c_int) :: ndims
    integer(c_int64_t), dimension(2) :: tensor_shape
    integer(c_int) :: dtype
    integer(c_int) :: device_type
    logical(c_bool) :: requires_grad

    ndims = 2
    tensor_shape = [2, 3]
    dtype = torch_kFloat32
    device_type = torch_kCPU
    requires_grad = .false.

    call torch_tensor_zeros(tensor, ndims, tensor_shape, dtype, device_type, &
                            requires_grad)

    ! Check if tensor is not null
    @assertTrue(tensor%p /= c_null_ptr)
  end subroutine test_torch_tensor_zeros

  @test
  subroutine test_torch_tensor_ones()
    type(torch_tensor) :: tensor
    integer(c_int) :: ndims
    integer(c_int64_t), dimension(2) :: tensor_shape
    integer(c_int) :: dtype
    integer(c_int) :: device_type
    logical(c_bool) :: requires_grad

    ndims = 2
    tensor_shape = [2, 3]
    dtype = torch_kFloat32
    device_type = torch_kCPU
    requires_grad = .false.

    call torch_tensor_ones(tensor, ndims, tensor_shape, dtype, device_type, &
                           requires_grad)

    ! Check if tensor is not null
    @assertTrue(tensor%p /= c_null_ptr)
  end subroutine test_torch_tensor_ones

end module test_ftorch
