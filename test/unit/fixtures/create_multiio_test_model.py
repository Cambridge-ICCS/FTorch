"""Create a neural network model with multiple inputs/outputs & save to TorchScript."""

import torch
from torch import nn


class MultiIONet(nn.Module):
    """PyTorch module multiplying two input vectors by 2 and 3, respectively."""

    def __init__(self) -> None:
        """
        Initialize the MultiIONet model.

        Consists of two Linear layers with weights predefined to
        multiply the inputs by 2 and 3, respectively.
        """
        super().__init__()
        self.linear1 = nn.Linear(4, 4, bias=False)
        self.linear2 = nn.Linear(4, 4, bias=False)
        with torch.inference_mode():
            self.linear1.weight = nn.Parameter(2.0 * torch.eye(4))
            self.linear2.weight = nn.Parameter(3.0 * torch.eye(4))

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
        a = self.linear1(batch1)
        b = self.linear2(batch2)
        return a, b


model = MultiIONet().eval()
scripted_model = torch.jit.script(model)
scripted_model.save("multiionet.pt")
