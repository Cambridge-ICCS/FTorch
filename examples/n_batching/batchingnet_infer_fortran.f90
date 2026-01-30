program batchingnet_infer

  use, intrinsic :: iso_fortran_env, only : sp => real32
  use ftorch, only : torch_model, torch_tensor, torch_kCPU, torch_delete, &
                    torch_tensor_from_array, torch_model_load, torch_model_forward
  use ftorch_test_utils, only : assert_allclose

  implicit none
  integer, parameter :: wp = sp
  integer :: num_args, ix
  character(len=128), dimension(:), allocatable :: args

  ! Single input and output
  real(wp), dimension(5), target :: in_data_single, out_data_single, expected_single
  type(torch_tensor), dimension(1) :: in_tensors_single, out_tensors_single

  ! Batched input and output
  real(wp), dimension(2,5), target :: in_data_batch, out_data_batch, expected_batch
  type(torch_tensor), dimension(1) :: in_tensors_batch, out_tensors_batch

  ! Multidimensional batched input and output
  real(wp), dimension(2,3,5), target :: in_data_multi, out_data_multi, expected_multi
  type(torch_tensor), dimension(1) :: in_tensors_multi, out_tensors_multi

  ! Bad example (mismatching batch dimensions) input and output
  real(wp), dimension(5,2), target :: in_data_bad, out_data_bad
  type(torch_tensor), dimension(1) :: in_tensors_bad, out_tensors_bad

  type(torch_model) :: model
  logical :: test_pass_single, test_pass_batch, test_pass_multi

  ! ===================================================================
  ! Load model from command line arg
  ! Note that this single loaded model will be used in all batching examples below.

  num_args = command_argument_count()
  allocate(args(num_args))
  do ix = 1, num_args
      call get_command_argument(ix,args(ix))
  end do

  ! Load model
  call torch_model_load(model, args(1), torch_kCPU)

  ! ===================================================================
  ! Single inference

  in_data_single = 1.0_wp
  call torch_tensor_from_array(in_tensors_single(1), in_data_single, torch_kCPU)
  call torch_tensor_from_array(out_tensors_single(1), out_data_single, torch_kCPU)

  call torch_model_forward(model, in_tensors_single, out_tensors_single)

  write (*,*) "--- Single input output: ---"
  write (*,*) out_data_single(:)
  write (*,*)

  expected_single = [0.0_wp, 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp]
  test_pass_single = assert_allclose(out_data_single, expected_single, &
                                    test_name="BatchingNet single", rtol=1e-5)

  call torch_delete(in_tensors_single)
  call torch_delete(out_tensors_single)

  ! ===================================================================
  ! Batched inference

  in_data_batch(1,:) = 1.0_wp
  in_data_batch(2,:) = 2.0_wp
  call torch_tensor_from_array(in_tensors_batch(1), in_data_batch, torch_kCPU)
  call torch_tensor_from_array(out_tensors_batch(1), out_data_batch, torch_kCPU)

  call torch_model_forward(model, in_tensors_batch, out_tensors_batch)

  write (*,*) "--- Batched input output: ---"
  write (*,*) out_data_batch(1,:)
  write (*,*) out_data_batch(2,:)
  write (*,*)

  expected_batch(1,:) = [0.0_wp, 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp]
  expected_batch(2,:) = [0.0_wp, 2.0_wp, 4.0_wp, 6.0_wp, 8.0_wp]
  test_pass_batch = assert_allclose(out_data_batch, expected_batch, &
                                   test_name="BatchingNet batch", rtol=1e-5)

  call torch_delete(in_tensors_batch)
  call torch_delete(out_tensors_batch)

  ! ===================================================================
  ! Multidimensional batched inference

  in_data_multi(1,1,:) = 1.0_wp
  in_data_multi(1,2,:) = 2.0_wp
  in_data_multi(1,3,:) = 3.0_wp
  in_data_multi(2,1,:) = 10.0_wp
  in_data_multi(2,2,:) = 20.0_wp
  in_data_multi(2,3,:) = 30.0_wp
  call torch_tensor_from_array(in_tensors_multi(1), in_data_multi, torch_kCPU)
  call torch_tensor_from_array(out_tensors_multi(1), out_data_multi, torch_kCPU)

  call torch_model_forward(model, in_tensors_multi, out_tensors_multi)

  write (*,*) "--- Multidimensional batched input/output: ---"
  write (*,*) "Input (1,1): ", in_data_multi(1,1,:)
  write (*,*) "Output (1,1):", out_data_multi(1,1,:)
  write (*,*) "Input (2,3): ", in_data_multi(2,3,:)
  write (*,*) "Output (2,3):", out_data_multi(2,3,:)
  write (*,*)

  expected_multi(1,1,:) = [0.0_wp, 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp]
  expected_multi(1,2,:) = [0.0_wp, 2.0_wp, 4.0_wp, 6.0_wp, 8.0_wp]
  expected_multi(1,3,:) = [0.0_wp, 3.0_wp, 6.0_wp, 9.0_wp, 12.0_wp]
  expected_multi(2,1,:) = [0.0_wp, 10.0_wp, 20.0_wp, 30.0_wp, 40.0_wp]
  expected_multi(2,2,:) = [0.0_wp, 20.0_wp, 40.0_wp, 60.0_wp, 80.0_wp]
  expected_multi(2,3,:) = [0.0_wp, 30.0_wp, 60.0_wp, 90.0_wp, 120.0_wp]
  test_pass_multi = assert_allclose(out_data_multi, expected_multi, &
                                   test_name="BatchingNet multidim batch", rtol=1e-5)

  call torch_delete(in_tensors_multi)
  call torch_delete(out_tensors_multi)

  ! ===================================================================
  ! Failing example with misaligned batching dimensions
  ! Uncomment the model forward call to see an error due to incorrect batching
  ! dimension order.
  ! Despite using pointers and a more fluid data layout, the standard approach to
  ! calling FTorch will still raise a matrix multiplication error if the batching
  ! dimensions mismatch, protecting against unforseen bugs.

  in_data_bad = 1.0_wp
  call torch_tensor_from_array(in_tensors_bad(1), in_data_bad, torch_kCPU)
  call torch_tensor_from_array(out_tensors_bad(1), out_data_bad, torch_kCPU)

  ! Forward call will will error:
  ! 5, the feature dimension expected by the net, should be last,
  ! preceded by any batching dimensions.
  ! call torch_model_forward(model, in_tensors_bad, out_tensors_bad)

  call torch_delete(in_tensors_bad)
  call torch_delete(out_tensors_bad)

  ! ===================================================================
  ! Cleanup
  call torch_delete(model)

  if (.not. (test_pass_single .and. test_pass_batch .and. test_pass_multi)) then
     stop 999
  end if

  write (*,*) "BatchingNet Fortran example ran successfully"

end program batchingnet_infer
