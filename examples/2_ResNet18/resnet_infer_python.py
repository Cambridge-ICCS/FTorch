"""Load ResNet-18 saved to TorchScript and run inference with an example image."""

from math import isclose
import numpy as np
import torch
from resnet18 import print_top_results


def deploy(saved_model: str, device: str, batch_size: int = 1) -> torch.Tensor:
    """
    Load TorchScript ResNet-18 and run inference with Tensor from example image.

    Parameters
    ----------
    saved_model : str
        location of ResNet-18 saved to Torchscript
    device : str
        Torch device to run model on, 'cpu' or 'cuda'
    batch_size : int
        batch size to run (default 1)

    Returns
    -------
    output : torch.Tensor
        result of running inference on model with Tensor of ones
    """
    transposed_shape = [224, 224, 3, batch_size]
    precision = np.float32

    np_data = np.fromfile("data/image_tensor.dat", dtype=precision)
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
    assert isclose(
        torch.max(torch.nn.functional.softmax(output[0], dim=0)),
        0.8846225142478943,
        abs_tol=1e-8,
    )


if __name__ == "__main__":
    saved_model_file = "saved_resnet18_model_cpu.pt"

    device_to_run = "cpu"
    # device_to_run = "cuda"

    batch_size_to_run = 1

    result = deploy(saved_model_file, device_to_run, batch_size_to_run)
    print_top_results(result)
    check_results(result)
