"""Load ResNet-18 saved to TorchScript and run inference with an example image."""

import os
from math import isclose

import numpy as np
import torch
from resnet18 import print_top_results


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
        Torch device to run model on, 'cpu' or 'cuda'
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

    np_data = np.fromfile(os.path.join(data_dir, "image_tensor.dat"), dtype=precision)
    np_data = np_data.reshape(transposed_shape)
    np_data = np_data.transpose()
    input_tensor = torch.from_numpy(np_data)

    if device == "cpu":
        # Load saved TorchScript model
        model = torch.jit.load(saved_model)
        # Inference
        output = model.forward(input_tensor)

    elif device == "cuda":
        # All previously saved modules, no matter their device, are first
        # loaded onto CPU, and then are moved to the devices they were saved
        # from, so we don't need to manually transfer the model to the GPU
        model = torch.jit.load(saved_model)
        input_tensor_gpu = input_tensor.to(torch.device("cuda"))
        output_gpu = model.forward(input_tensor_gpu)
        output = output_gpu.to(torch.device("cpu"))

    else:
        device_error = f"Device '{device}' not recognised."
        raise ValueError(device_error)

    return output


def check_results(output: torch.Tensor) -> None:
    """
    Compare top model output to expected result.

    Parameters
    ----------
    output: torch.Tensor
        Output from ResNet-18.
    """
    #  Run a softmax to get probabilities
    predicted_prob = torch.max(torch.nn.functional.softmax(output[0], dim=0))
    expected_prob = 0.8846225142478943
    if not isclose(predicted_prob, expected_prob, abs_tol=1e-5):
        result_error = (
            f"Predicted probability: {predicted_prob} does not match the expected"
            f" value: {expected_prob}."
        )
        raise ValueError(result_error)


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
    parsed_args = parser.parse_args()
    saved_model_file = os.path.join(parsed_args.filepath, "saved_resnet18_model_cpu.pt")
    data_dir = parsed_args.data_dir

    device_to_run = "cpu"

    batch_size_to_run = 1

    with torch.inference_mode():
        result = deploy(saved_model_file, device_to_run, data_dir, batch_size_to_run)
    print_top_results(result, data_dir)
    check_results(result)
