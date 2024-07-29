#!/usr/bin/env python3
"""Load saved MultIONet to TorchScript and run inference example."""

import os
import sys
import torch


def deploy(saved_model: str, device: str, batch_size: int = 1) -> torch.Tensor:
    """
    Load TorchScript MultiIONet and run inference with example Tensors.

    Parameters
    ----------
    saved_model : str
        location of MultiIONet model saved to Torchscript
    device : str
        Torch device to run model on, 'cpu' or 'cuda'
    batch_size : int
        batch size to run (default 1)

    Returns
    -------
    outputs : tuple
        result of running inference on model with example Tensor inputs
    """
    input_tensors = (
        torch.tensor([0.0, 1.0, 2.0, 3.0]).repeat(batch_size, 1),
        torch.tensor([0.0, -1.0, -2.0, -3.0]).repeat(batch_size, 1),
    )

    if device == "cpu":
        # Load saved TorchScript model
        model = torch.jit.load(saved_model)
        # Inference
        outputs = model.forward(*input_tensors)

    elif device == "cuda":
        # All previously saved modules, no matter their device, are first
        # loaded onto CPU, and then are moved to the devices they were saved
        # from, so we don't need to manually transfer the model to the GPU
        model = torch.jit.load(saved_model)
        input_tensors_gpu = input_tensors.to(torch.device("cuda"))
        outputs_gpu = model.forward(*input_tensors_gpu)
        outputs = outputs_gpu.to(torch.device("cpu"))

    else:
        raise ValueError(f"Device '{device}' not recognised.")

    return outputs


if __name__ == "__main__":
    device_type = "cpu"
    # device_type = "cuda"

    filepath = os.path.dirname(__file__) if len(sys.argv) == 1 else sys.argv[1]
    filename = f"saved_multiio_model_{device_type}.pt"
    saved_model_file = os.path.join(filepath, filename)

    batch_size_to_run = 1

    with torch.no_grad():
        result = deploy(saved_model_file, device_type, batch_size_to_run)

    print(result)
