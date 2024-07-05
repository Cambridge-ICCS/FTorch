#!/bin/env python3
"""Module defining a simple PyTorch 'Net' for coupling to Fortran."""

import torch
from torch import nn


class SimpleNet(nn.Module):
    """PyTorch module multiplying an input vector by 2."""

    def __init__(
        self,
    ) -> None:
        """
        Initialize the SimpleNet model.

        Consists of a single Linear layer with weights predefined to
        multiply the input by 2.
        """
        super().__init__()
        self._fwd_seq = nn.Sequential(
            nn.Linear(5, 5, bias=False),
        )
        with torch.no_grad():
            self._fwd_seq[0].weight = nn.Parameter(2.0 * torch.eye(5))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Pass ``batch`` through the model.

        Parameters
        ----------
        batch : torch.Tensor
            A mini-batch of input vectors of length 5.

        Returns
        -------
        torch.Tensor
            batch scaled by 2.

        """
        return self._fwd_seq(batch)


if __name__ == "__main__":

    # Check CUDA is supported on the hardware
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    device_name = torch.cuda.current_device()
    print(f"Using CUDA device '{torch.cuda.get_device_name(device_name)}'")
    device = torch.device("cuda")

    # Put the model on the GPU
    model = SimpleNet()
    model.eval()
    model.to(device)

    # Put the input tensor on the GPU
    input_tensor = torch.Tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    input_tensor_gpu = input_tensor.to(device)

    # Run inference
    with torch.no_grad():
        print(model(input_tensor_gpu))
