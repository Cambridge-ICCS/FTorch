program gpu_tensor_pointer

  ! This example checks whether it is possible to construct a `torch_tensor`
  ! directly on the GPU (i.e. without a host-to-device copy) by passing a GPU
  ! device pointer to `torch_tensor_from_blob`. This contrasts with the two
  ! approaches normally used in FTorch: creating tensors on the CPU and moving
  ! them to the GPU with `torch_tensor_to` (or equivalently specifying a GPU
  ! device in one of the standard constructors such as `torch_tensor_ones`),
  ! and is motivated by codes in which data already lives on the GPU.
  !
  ! The array is defined using CUDA Fortran and a device pointer to it is
  ! obtained with `c_devloc` from the `iso_c_binding` module.
  !
  ! Notes:
  !  * `torch_tensor_from_array` cannot be used directly with CUDA Fortran
  !    arrays because it takes a regular Fortran data pointer, whereas `c_loc`
  !    cannot be applied to a `device` variable. Instead, `torch_tensor_from_blob`
  !    is used and the shape and strides are supplied explicitly, replicating
  !    what `torch_tensor_from_array` does internally (contiguous, column-major
  !    order i.e. stride 1 in the leading dimension).
  !  * When inspecting `torch_from_blob` in `src/ctorch.cpp` it will be seen
  !    that `torch::from_blob` is called with CPU `TensorOptions`, with the GPU
  !    device only applied afterwards via `.to(device)`. It is therefore not
  !    obvious a priori whether a device pointer is respected (libtorch infers
  !    the device of the data from its placement in memory) or whether host
  !    memory is assumed (which would lead to a crash or garbage values).
  !    Compiling and running this example answers that question.
  !
  ! The check proceeds in two stages:
  !  1. constructing the tensor, interrogating its device/shape/strides and
  !     printing it - this can signal basic success or crash outright;
  !  2. a quantitative check that GPU compute performed on the tensor produces
  !     the correct result in GPU memory - performed by assigning the result
  !     into an output tensor that views a second GPU array, which is
  !     subsequently copied back to the host and compared against the expected
  !     result in pure Fortran.
  !
  ! NOTE: This example is designed to be compiled and run on a machine with an
  ! NVIDIA GPU, using a CUDA Fortran capable compiler (e.g. nvfortran) and an
  ! FTorch installation built with `-DGPU_DEVICE=CUDA`.

  ! CUDA Fortran modules
  use cudafor, only: c_devloc
  use, intrinsic :: iso_c_binding, only: c_int, c_ptr
  use, intrinsic :: iso_fortran_env, only: int32, int64, sp => real32

  ! FTorch modules
  use ftorch, only: assignment(=), operator(+), torch_kCUDA, torch_kFloat32,     &
                    torch_tensor, torch_tensor_delete, torch_tensor_from_blob, &
                    torch_tensor_print
  use ftorch_test_utils, only: allclose

  implicit none

  ! Set working precision for reals to be 32-bit floats
  integer, parameter :: wp = sp

  ! Define two 2D arrays to live on the GPU and their host-side counterparts.
  ! `in_device` will be wrapped by a torch_tensor constructed from its device
  ! pointer, while `out_device` will be wrapped by an output tensor that the
  ! result of a GPU computation is assigned into so the result can be checked.
  integer, parameter :: nx = 2, ny = 3
  real(wp), device, allocatable :: in_device(:,:), out_device(:,:)
  real(wp), allocatable :: out_host(:,:), expected(:,:)
  integer :: i

  ! Variables for constructing tensors with torch_tensor_from_blob
  type(torch_tensor) :: a, b
  integer(int32), parameter :: ndims = 2
  integer(int64) :: tensor_shape(ndims), tensor_strides(ndims)
  type(c_ptr) :: in_device_ptr, out_device_ptr

  ! Flag for testing
  logical :: test_pass

  ! Allocate the arrays on the GPU and on the host.
  ! Note that the expected result is computed in pure Fortran on the host
  ! so it remains valid even if libtorch cannot access the device pointer.
  allocate(in_device(nx, ny), out_device(nx, ny))
  allocate(out_host(nx, ny), expected(nx, ny))

  ! Fill the input GPU array with small integers (so any corruption of the
  ! values is easy to spot). The assignment performs a host -> device copy.
  in_device = reshape([(real(i, wp), i = 1, nx * ny)], [nx, ny])
  expected = in_device + in_device

  ! Compute the shape and stride information describing the arrays, replicating
  ! what `torch_tensor_from_array` computes from its array argument.
  ! For a contiguous Fortran array this is column-major order i.e. strides
  ! [1, nx] for an nx x ny array.
  tensor_shape = shape(in_device, kind=int64)
  tensor_strides(1) = 1_int64
  do i = 2, ndims
    tensor_strides(i) = tensor_strides(i - 1) * tensor_shape(i - 1)
  end do

  ! Get the device pointers to the GPU arrays using c_devloc
  in_device_ptr = c_devloc(in_device)
  out_device_ptr = c_devloc(out_device)

  ! ----------------------------------------------------------------------
  ! Stage 1: construct a torch_tensor wrapping the GPU memory directly
  ! ----------------------------------------------------------------------
  write(*,*) "Stage 1: constructing a torch_tensor from a GPU device pointer..."
  call torch_tensor_from_blob(a, in_device_ptr, ndims, tensor_shape, tensor_strides, &
                              torch_kFloat32, torch_kCUDA, device_index=0)
  write(*,*) "  ... torch_tensor object constructed."

  ! Interrogate the tensor: its device indicates whether the data was wrapped
  ! in-place on the GPU (expected if device pointers are supported), or whether
  ! the data was copied from host memory (in which case this code may also
  ! crash before reaching here)
  write(*,*) "Device type of constructed tensor:", a%device_type(), &
             " (expected:", torch_kCUDA, " i.e. torch_kCUDA)"
  write(*,*) "Device index of constructed tensor:", a%device_index(), " (expected: 0)"
  write(*,*) "Shape of constructed tensor:", a%shape()     ! Expected: nx ny
  write(*,*) "Stride of constructed tensor:", a%stride()   ! Expected: 1 nx

  ! Inspecting the values provides a stronger check still: printing a tensor
  ! that mistakenly wraps host memory (or a device pointer treated as such)
  ! will produce garbage or crash
  write(*,*) "Contents of tensor constructed from a GPU device pointer:"
  call torch_tensor_print(a)
  write(*,*)

  ! ----------------------------------------------------------------------
  ! Stage 2: verify GPU compute on the wrapped tensor produces the correct
  ! result in GPU memory
  ! ----------------------------------------------------------------------
  write(*,*) "Stage 2: performing GPU arithmetic on the constructed tensor..."

  ! Create a torch_tensor from the output GPU array in the same way. Setting
  ! the output tensor up in advance means the result of the arithmetic below is
  ! written directly into the GPU array, from where it can be copied back to
  ! the host and verified
  call torch_tensor_from_blob(b, out_device_ptr, ndims, tensor_shape, tensor_strides, &
                              torch_kFloat32, torch_kCUDA, device_index=0)

  ! Perform arithmetic on the tensors using the overloaded addition operator.
  ! This should be computed on the GPU and the resulting values written into
  ! the out_device array.
  b = a + a

  ! Copy the result back from the device array and compare with the expected
  ! result computed on the host
  out_host = out_device
  test_pass = allclose(out_host, expected, test_name="gpu_tensor_pointer arithmetic")
  if (.not. test_pass) then
    write(*,*) "  ... GPU arithmetic gave unexpected values:"
    write(*,*) "      got      =", out_host
    write(*,*) "      expected =", expected
    stop 1
  end if
  write(*,*) "  ... GPU arithmetic gave the expected values."
  write(*,*)

  ! Clean up
  ! --------
  call torch_tensor_delete(a)
  call torch_tensor_delete(b)
  deallocate(in_device, out_device)
  deallocate(out_host, expected)

  write(*,*) "GPU device pointer example ran successfully"

end program gpu_tensor_pointer
