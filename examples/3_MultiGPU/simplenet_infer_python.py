"""Load saved SimpleNet to TorchScript and run inference example."""

import torch
from mpi4py import MPI


def deploy(saved_model: str, device: str, batch_size: int = 1) -> torch.Tensor:
    """
    Load TorchScript SimpleNet and run inference with example Tensor.

    Parameters
    ----------
    saved_model : str
        location of SimpleNet model saved to Torchscript
    device : str
        Torch device to run model on, 'cpu' or 'cuda'
    batch_size : int
        batch size to run (default 1)

    Returns
    -------
    output : torch.Tensor
        result of running inference on model with example Tensor input
    """
    input_tensor = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]).repeat(batch_size, 1)

    # Add the rank (device index) to each tensor to make them differ
    input_tensor += MPI.COMM_WORLD.rank

    if device == "cpu":
        # Load saved TorchScript model
        model = torch.jit.load(saved_model)
        # Inference
        output = model.forward(input_tensor)

    elif device.startswith("cuda"):
        # All previously saved modules, no matter their device, are first
        # loaded onto CPU, and then are moved to the devices they were saved
        # from, so we don't need to manually transfer the model to the GPU
        model = torch.jit.load(saved_model)
        input_tensor_gpu = input_tensor.to(torch.device(device))
        output_gpu = model.forward(input_tensor_gpu)
        output = output_gpu.to(torch.device("cpu"))

    else:
        device_error = f"Device '{device}' not recognised."
        raise ValueError(device_error)

    return output


if __name__ == "__main__":
    saved_model_file = "saved_simplenet_model_cuda.pt"

    rank = MPI.COMM_WORLD.rank
    device_to_run = f"cuda:{rank}"

    batch_size_to_run = 1

    with torch.no_grad():
        result = deploy(saved_model_file, device_to_run, batch_size_to_run)

    print(f"{rank}: {result}")
