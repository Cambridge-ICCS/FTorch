"""Module defining a simple PyTorch 'Net' for coupling to Fortran."""

import torch
from torch import nn


class MultiIONet(nn.Module):
    """PyTorch module multiplying two input vectors by 2 and 3, respectively."""

    def __init__(
        self,
    ) -> None:
        """
        Initialize the MultiIONet model.

        Consists of a single Linear layer with weights predefined to
        multiply the inputs by 2 and 3, respectively.
        """
        super().__init__()
        self._fwd_seq = nn.Sequential(nn.Linear(4, 4, bias=False))
        with torch.no_grad():
            self._fwd_seq[0].weight = nn.Parameter(
                torch.kron(torch.Tensor([[2.0, 0.0], [0.0, 3.0]]), torch.eye(4))
            )

    def forward(self, batch1: torch.Tensor, batch2: torch.Tensor):
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
        batch = torch.cat((batch1, batch2), dim=0).flatten()
        a, b = self._fwd_seq(batch).split(4)
        return a, b


if __name__ == "__main__":
    model = MultiIONet()
    model.eval()

    input_tensors = (
        torch.Tensor([0.0, 1.0, 2.0, 3.0]),
        torch.Tensor([-0.0, -1.0, -2.0, -3.0]),
    )

    with torch.no_grad():
        output_tensors = model(*input_tensors)

    print(output_tensors)

    for output_i, input_i, scale_factor in zip(output_tensors, input_tensors, (2, 3)):
        if not torch.allclose(output_i, scale_factor * input_i):
            result_error = (
                f"result:\n{output_i}\ndoes not match expected value:\n"
                f"{scale_factor * input_i}"
            )
            raise ValueError(result_error)
