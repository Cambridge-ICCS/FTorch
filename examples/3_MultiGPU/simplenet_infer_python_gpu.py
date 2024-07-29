#!/bin/env python3
"""Load saved SimpleNet to TorchScript and run inference example."""

from mpi4py import MPI
import os
import sys
import torch


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
        model.to(torch.device(device))
        input_tensor_gpu = input_tensor.to(torch.device(device))
        output_gpu = model.forward(input_tensor_gpu)
        output = output_gpu.to(torch.device("cpu"))

    else:
        raise ValueError(f"Device '{device}' not recognised.")

    return output


if __name__ == "__main__":

    # Check CUDA is supported on the hardware
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    device_name = torch.cuda.current_device()
    print(f"Using CUDA device '{torch.cuda.get_device_name(device_name)}'")

    device_type = "cuda"
    device_index = 0
    device_to_run = f"{device_type}:{device_index}"

    filepath = os.path.dirname(__file__) if len(sys.argv) == 1 else sys.argv[1]
    filename = f"saved_simplenet_model_{device_type}.pt"
    saved_model_file = os.path.join(filepath, filename)

    batch_size_to_run = 1

    with torch.no_grad():
        result = deploy(saved_model_file, device_to_run, batch_size_to_run)

    print(f"{MPI.COMM_WORLD.rank}: {result}")
