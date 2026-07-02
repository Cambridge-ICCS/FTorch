"""Load ResNet-18 saved to TorchScript and run inference with an example image."""

import os

import numpy as np
import torch
from resnet18 import check_results, print_top_results


def deploy(
    saved_model: str, device: str, data_dir: str, batch_size: int = 1
) -> torch.Tensor:
    """
    Load TorchScript ResNet-18 and run inference with Tensor from example image.

    Parameters
    ----------
    saved_model : str
        location of ResNet-18 saved to Torchscript
    device : str
        Torch device to run model on, e.g., 'cpu' or 'cuda'
    data_dir : str
        Path to data directory
    batch_size : int
        batch size to run (default 1)

    Returns
    -------
    output : torch.Tensor
        result of running inference on model with Tensor of ones
    """
    transposed_shape = [224, 224, 3, batch_size]
    precision = np.float32

    # Setup input tensor
    np_data = np.fromfile(os.path.join(data_dir, "image_tensor.dat"), dtype=precision)
    np_data = np_data.reshape(transposed_shape)
    np_data = np_data.transpose()
    input_tensor = torch.from_numpy(np_data)

    # Load saved TorchScript model
    model = torch.jit.load(saved_model)

    # Propagate
    output = model.forward(input_tensor).to("cpu")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--filepath",
        help="Path to the file containing the PyTorch model",
        type=str,
        default=os.path.dirname(__file__),
    )
    parser.add_argument(
        "--data_dir",
        help="Path to the directory containing the input data",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data"),
    )
    parser.add_argument(
        "--device_type",
        help="Device type to run the inference on",
        type=str,
        choices=["cpu", "cuda", "hip", "xpu", "mps"],
        default="cpu",
    )
    parsed_args = parser.parse_args()
    filepath = parsed_args.filepath
    data_dir = parsed_args.data_dir
    device_type = parsed_args.device_type
    saved_model_file = os.path.join(
        filepath, f"torchscript_resnet18_model_{device_type}.pt"
    )
    batch_size_to_run = 1

    with torch.inference_mode():
        result = deploy(saved_model_file, device_type, data_dir, batch_size_to_run)
    print_top_results(result, data_dir)
    check_results(result)
