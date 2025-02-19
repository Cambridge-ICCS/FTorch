"""Load saved SimpleNet to TorchScript and run inference example."""

import os

import torch


def deploy(saved_model: str, device: str, batch_size: int = 1) -> torch.Tensor:
    """
    Load TorchScript SimpleNet and run inference with example Tensor.

    Parameters
    ----------
    saved_model : str
        location of SimpleNet model saved to Torchscript
    device : str
        Torch device to run model on, 'cpu' or 'cuda'. May be followed by a colon and
        then a device index, e.g., 'cuda:0' for the 0th CUDA device.
    batch_size : int
        batch size to run (default 1)

    Returns
    -------
    output : torch.Tensor
        result of running inference on model with example Tensor input
    """
    input_tensor = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]).repeat(batch_size, 1)

    if device == "cpu":
        # Load saved TorchScript model
        model = torch.jit.load(saved_model)
        # Inference
        return model.forward(input_tensor)

    if device.startswith("cuda"):
        pass
    elif device.startswith("xpu"):
        # XPU devices need to be initialised before use
        torch.xpu.init()
    elif device.startswith("mps"):
        pass
    else:
        device_error = f"Device '{device}' not recognised."
        raise ValueError(device_error)

    # Add the device index to each tensor to make them differ
    input_tensor += int(device.split(":")[-1] or 0)

    # All previously saved modules, no matter their device, are first
    # loaded onto CPU, and then are moved to the devices they were saved
    # from.
    # Since we are loading one saved model to multiple devices we explicitly
    # transfer using `.to(device)` to ensure the model is on the correct index.
    model = torch.jit.load(saved_model)
    model = model.to(device)
    input_tensor_gpu = input_tensor.to(torch.device(device))
    output_gpu = model.forward(input_tensor_gpu)
    output = output_gpu.to(torch.device("cpu"))

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
        "--device_type",
        help="Device type to run the inference on",
        type=str,
        choices=["cpu", "cuda", "xpu", "mps"],
        default="cuda",
    )
    parsed_args = parser.parse_args()
    filepath = parsed_args.filepath
    device_type = parsed_args.device_type
    saved_model_file = os.path.join(filepath, f"saved_multigpu_model_{device_type}.pt")

    batch_size_to_run = 1

    # Use 2 devices unless MPS for which there is only one
    num_devices = 1 if device_type == "mps" else 2

    for device_index in range(num_devices):
        device_to_run = f"{device_type}:{device_index}"

        with torch.no_grad():
            result = deploy(saved_model_file, device_to_run, batch_size_to_run)

        print(f"Output on device {device_to_run}: {result}")

        expected = torch.Tensor([2 * (i + device_index) for i in range(5)])
        if not torch.allclose(result, expected):
            result_error = (
                f"result:\n{result}\ndoes not match expected value:\n{expected}"
            )
            raise ValueError(result_error)
