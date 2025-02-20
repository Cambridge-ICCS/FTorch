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
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device_type",
        help="Device type to run the inference on",
        type=str,
        choices=["cpu", "cuda", "xpu", "mps"],
        default="cuda",
    )
    parsed_args = parser.parse_args()
    device_type = parsed_args.device_type

    model = SimpleNet().to(torch.device(device_type))
    model.eval()

    input_tensor = torch.Tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    input_tensor_gpu = input_tensor.to(torch.device(device_type))

    print(
        f"SimpleNet forward pass on {device_type.capitalize()} device"
        f" {input_tensor_gpu.get_device()}"
    )
    with torch.no_grad():
        output_tensor = model(input_tensor_gpu).to("cpu")

    print(output_tensor)
    if not torch.allclose(output_tensor, 2 * input_tensor):
        result_error = (
            f"result:\n{output_tensor}\ndoes not match expected value:\n"
            f"{2 * input_tensor}"
        )
        raise ValueError(result_error)
