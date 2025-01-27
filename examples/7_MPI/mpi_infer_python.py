"""Load saved SimpleNet to TorchScript and run inference example."""

import os
import sys

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

    elif device == "cuda":
        # All previously saved modules, no matter their device, are first
        # loaded onto CPU, and then are moved to the devices they were saved
        # from, so we don't need to manually transfer the model to the GPU
        model = torch.jit.load(saved_model)
        input_tensor_gpu = input_tensor.to(torch.device("cuda"))
        output_gpu = model.forward(input_tensor_gpu)
        output = output_gpu.to(torch.device("cpu"))

    else:
        device_error = f"Device '{device}' not recognised."
        raise ValueError(device_error)

    return output


if __name__ == "__main__":
    filepath = os.path.dirname(__file__) if len(sys.argv) == 1 else sys.argv[1]
    saved_model_file = os.path.join(filepath, "saved_simplenet_model_cpu.pt")

    comm = MPI.COMM_WORLD
    rank = comm.rank
    device_to_run = "cpu"

    batch_size_to_run = 1

    # Run inference on each rank
    with torch.no_grad():
        result = deploy(saved_model_file, device_to_run, batch_size_to_run)
    print(f"rank {rank}: result:\n{result}")

    # Gather the outputs onto rank 0
    recvbuf = torch.empty([comm.size, 5], dtype=torch.float32) if rank == 0 else None
    comm.Gather(result, recvbuf, root=0)

    # Check that the correct values were attained
    if rank == 0:
        for rank_chk, result_chk in enumerate(recvbuf):
            expected = torch.Tensor([2 * (i + rank_chk) for i in range(5)])
            if not torch.allclose(expected, result_chk):
                result_error = (
                    f"rank {rank_chk}: result:\n{result_chk}\n"
                    f"does not match expected value:\n{expected}"
                )
                raise ValueError(result_error)

        print("MPI Python example ran successfully")
