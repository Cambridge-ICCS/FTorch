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
    import argparse

    # Parse user input
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device_type",
        help="Device type to run the inference on",
        type=str,
        choices=["cpu", "cuda", "hip", "xpu", "mps"],
        default="cpu",
    )
    parsed_args = parser.parse_args()
    device_type = parsed_args.device_type

    # Construct an instance of the BatchingNet model on the specified device
    model = BatchingNet().to(device_type)
    model.eval()

    # Save the model in PyTorch format
    torch.save(model.state_dict(), f"saved_batchingnet_model_{device_type}.pt")

    # Create an arbitrary input tensor and save it in PyTorch format
    input_tensor = torch.ones(5).to(device_type)
    torch.save(input_tensor, f"saved_batchingnet_input_tensor_{device_type}.pt")

    # Propagate the input tensor through the model
    with torch.inference_mode():
        output_tensor = model(input_tensor)
    print(f"Model output: {output_tensor}")

    # Perform a basic check of the model output
    if not torch.allclose(output_tensor, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])):
        result_error = (
            f"result:\n{output_tensor}\ndoes not match expected value:\n"
            f"{2 * input_tensor}"
        )
        raise ValueError(result_error)
