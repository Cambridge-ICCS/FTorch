"""Module defining a simple PyTorch 'Net' for coupling to Fortran."""

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

    # Construct an instance of the SimpleNet model on the specified device
    model = MultiIONet().to(device_type)
    model.eval()

    # Save the model in PyTorch format
    torch.save(model.state_dict(), f"saved_multiio_model_{device_type}.pt")

    # Create an arbitrary input tensor and save it in PyTorch format
    input_tensors = (
        torch.Tensor([0.0, 1.0, 2.0, 3.0]).to(device_type),
        torch.Tensor([-0.0, -1.0, -2.0, -3.0]).to(device_type),
    )
    torch.save(input_tensors, f"saved_multiio_input_tensor_{device_type}.pt")

    # Propagate the input tensor through the model
    with torch.inference_mode():
        output_tensors = model(*input_tensors)
    print(f"Model output: {output_tensors}")

    # Perform a basic check of the model output
    for output_i, input_i, scale_factor in zip(output_tensors, input_tensors, (2, 3)):
        if not torch.allclose(output_i, scale_factor * input_i):
            result_error = (
                f"result:\n{output_i}\ndoes not match expected value:\n"
                f"{scale_factor * input_i}"
            )
            raise ValueError(result_error)
