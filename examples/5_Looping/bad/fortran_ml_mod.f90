module ml_mod

  use, intrinsic :: iso_fortran_env, only : sp=>real32

  ! Import our library for interfacing with PyTorch
  use ftorch, only : torch_tensor, torch_model, torch_kCPU, torch_delete, &
                     torch_tensor_from_array, torch_model_load, torch_model_forward

  implicit none

  private
  public ml_routine

  ! Set working precision for reals
  integer, parameter :: wp = sp

  contains

  subroutine ml_routine(in_data, out_data)

    ! Set up Fortran data structures
    real(wp), dimension(5), target, intent(in)  :: in_data
    real(wp), dimension(5), target, intent(out) :: out_data

    ! Set up Torch data structures
    ! The net, a vector of input tensors, and a vector of output tensors
    integer, parameter, dimension(1) :: tensor_layout = [1]
    type(torch_tensor), dimension(1) :: input_tensors
    type(torch_tensor), dimension(1) :: output_tensors
    type(torch_model) :: torch_net

    ! Get TorchScript model file
    character(len=128) :: model_torchscript_file

    ! Create Torch input/output tensors from the above arrays
    call torch_tensor_from_array(input_tensors(1), in_data, tensor_layout, torch_kCPU)
    call torch_tensor_from_array(output_tensors(1), out_data, tensor_layout, torch_kCPU)

    ! Load ML model
    model_torchscript_file = '../saved_simplenet_model.pt'
    call torch_model_load(torch_net, model_torchscript_file, torch_kCPU)

    ! Infer
    call torch_model_forward(torch_net, input_tensors, output_tensors)

    ! Cleanup
    call torch_delete(input_tensors)
    call torch_delete(output_tensors)
    call torch_delete(torch_net)

  end subroutine ml_routine

end module ml_mod
