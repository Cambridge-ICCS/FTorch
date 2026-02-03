"""
Module defining a PyTorch 'BatchingNet' for clear batching demonstration.

This net multiplies each input feature by a different value, making the effect of
batching and the network operation obvious in the outputs.
"""

import torch
from torch import nn


class BatchingNet(nn.Module):
    """
    PyTorch module multiplying each input feature by a distinct scalar.

    The model consists of a single Linear layer with weights set to [0, 1, 2, 3, 4].
    This makes the effect of batching and the network operation clear.
    """

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


if __name__ == "__main__":
    model = BatchingNet()
    model.eval()
    input_tensor = torch.ones(5)
    with torch.inference_mode():
        output_tensor = model(input_tensor)

    print(output_tensor)
    if not torch.allclose(output_tensor, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])):
        result_error = (
            f"result:\n{output_tensor}\ndoes not match expected value:\n"
            f"{2 * input_tensor}"
        )
        raise ValueError(result_error)
