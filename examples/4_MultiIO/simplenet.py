#!/usr/bin/env python3
"""Module defining a simple PyTorch 'Net' for coupling to Fortran."""

import torch
from torch import nn


class SimpleNet(nn.Module):
    """PyTorch module multiplying two input vectors by 2 and 3, respectively."""

    def __init__(
        self,
    ) -> None:
        """
        Initialize the SimpleNet model.

        Consists of a single Linear layer with weights predefined to
        multiply the inputs by 2 and 3, respectively.
        """
        super().__init__()
        self._fwd_seq = nn.Sequential(nn.Linear(10, 10, bias=False))
        with torch.no_grad():
            self._fwd_seq[0].weight = nn.Parameter(
                torch.kron(torch.Tensor([[2.0, 0.0], [0.0, 3.0]]), torch.eye(4))
            )

    def forward(self, batch1: torch.Tensor, batch2: torch.Tensor) -> torch.Tensor:
        """
        Pass ``batch1`` and ``batch2`` through the model.

        Parameters
        ----------
        batch1 : torch.Tensor
            A mini-batch of input vectors of length 4.
        batch2 : torch.Tensor
            Another mini-batch of input vectors of length 4.

        Returns
        -------
        torch.Tensor
            first batch scaled by 2.
        torch.Tensor
            second batch scaled by 3.

        """
        batch = torch.cat((batch1, batch2), dim=0)
        return self._fwd_seq(batch).view((2, 4))


if __name__ == "__main__":
    model = SimpleNet()
    model.eval()

    with torch.no_grad():
        print(
            model(
                torch.Tensor([0.0, 1.0, 2.0, 3.0]),
                torch.Tensor([-0.0, -1.0, -2.0, -3.0]),
            )
        )
