"""Load saved FNO1d to TorchScript and run inference example."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from fno1d_train import generate_sine_data


def deploy(saved_model: str, device: str, batch_size: int = 1) -> torch.Tensor:
    """
    Load TorchScript FNO1d and run inference with example Tensor.

    Parameters
    ----------
    saved_model : str
        location of FNO1d model saved to Torchscript
    device : str
        Torch device to run model on, 'cpu' or 'cuda'
    batch_size : int
        batch size to run (default 1)

    Returns
    -------
    output : torch.Tensor
        result of running inference on model with example Tensor input
    """
    input_tensor, grid_tensor, target_tensor = generate_sine_data(
        batch_size=batch_size, size_x=32
    )

    input_example = torch.cat((input_tensor, grid_tensor), dim=-1)
    print("Input example shape:", input_example.shape)

    if device == "cpu":
        # Load saved TorchScript model
        model = torch.jit.load(saved_model)
        # Inference
        output = model(input_example)

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

    return output, grid_tensor, target_tensor


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
    parsed_args = parser.parse_args()
    filepath = parsed_args.filepath
    saved_model_file = os.path.join(filepath, "saved_fno1d_model_cpu.pt")

    device_to_run = "cpu"

    batch_size_to_run = 1

    with torch.no_grad():
        predicted, grid_tensor, target_tensor = deploy(
            saved_model_file, device_to_run, batch_size_to_run
        )

    pred_vals = predicted.squeeze().numpy()

    # Compute absolute error
    error = np.abs(pred_vals - target_tensor.squeeze().numpy())

    print(error * 1000)
    # Total error
    total_error = np.sum(error)

    tol_sum = 0.2
    # Check against tolerance
    if total_error < tol_sum:
        print(f"Total absolute error = {total_error:.6f} — within tolerance.")
    else:
        print(f"Total absolute error = {total_error:.6f} — exceeds tolerance!")

    plt.plot(
        grid_tensor.squeeze().numpy(),
        target_tensor.squeeze().numpy(),
        label="True sin(2πx)",
    )
    plt.scatter(grid_tensor.squeeze().numpy(), pred_vals, label="Predicted")
    plt.legend()
    plt.title("Sine Wave Prediction")
    plt.savefig("train_fno1d_infer.png")
