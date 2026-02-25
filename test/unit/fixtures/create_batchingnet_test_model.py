"""Create a batching neural network model and write it out using TorchScript."""

import torch
from torch import nn


class BatchingNet(nn.Module):
    """PyTorch module multiplying each input feature by a distinct scalar."""

    def __init__(self) -> None:
        """
        Initialise the BatchingNet model.

        The model contains a single Linear layer (5 -> 5, no bias) with weights
        initialised so that output[i] = input[i] * i for i in 0..4.
        """
        super().__init__()
        self._fwd_seq = nn.Sequential(
            nn.Linear(5, 5, bias=False),
        )
        with torch.inference_mode():
            self._fwd_seq[0].weight = nn.Parameter(
                torch.diag(torch.arange(5, dtype=torch.float32))
            )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.

        Parameters
        ----------
        batch : torch.Tensor
            Input tensor of shape (..., 5). Can be batched or higher-dimensional.

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as input, scaled appropriately.
        """
        return self._fwd_seq(batch)


model = BatchingNet().eval()
scripted_model = torch.jit.script(model)
scripted_model.save("batchingnet.pt")
