module ml_mod

  use, intrinsic :: iso_fortran_env, only : sp=>real32

  ! Import our library for interfacing with PyTorch
  use ftorch

  implicit none

  private
  public ml_init, ml_routine, ml_final

  ! Set working precision for reals
  integer, parameter :: wp = sp
   
  ! Set up Torch data structures
  ! The net, a vector of input tensors, and a vector of output tensors
  integer, dimension(1) :: tensor_layout = [1]
  type(torch_tensor), dimension(1) :: input_tensors
  type(torch_tensor), dimension(1) :: output_tensors
  type(torch_model) :: torch_net
 
  ! Get TorchScript model file
  character(len=128) :: model_torchscript_file
 
  contains

  subroutine ml_init()

    ! Load ML model
    model_torchscript_file = 'saved_model.pt'
    call torch_model_load(torch_net, model_torchscript_file)
 
  end subroutine ml_init

  subroutine ml_routine(in_data, out_data)

    ! Set up Fortran data structures
    real(wp), dimension(5), target, intent(in)  :: in_data
    real(wp), dimension(5), target, intent(out) :: out_data
 
    ! Create Torch input/output tensors from the above arrays
    call torch_tensor_from_array(input_tensors(1), in_data, tensor_layout, torch_kCPU)
    call torch_tensor_from_array(output_tensors(1), out_data, tensor_layout, torch_kCPU)
 
    ! Infer
    call torch_model_forward(torch_net, input_tensors, output_tensors)
 
  end subroutine ml_routine

  subroutine ml_final()
    ! Cleanup
    call torch_delete(input_tensors)
    call torch_delete(output_tensors)
    call torch_delete(torch_net)

  end subroutine ml_final

end module ml_mod
