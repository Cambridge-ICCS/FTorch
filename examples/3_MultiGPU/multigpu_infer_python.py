"""Load saved SimpleNet to TorchScript and run inference example."""

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
    else:
        device_error = f"Device '{device}' not recognised."
        raise ValueError(device_error)

    # Add the device index to each tensor to make them differ
    input_tensor += int(device.split(":")[-1] or 0)

    # All previously saved modules, no matter their device, are first
    # loaded onto CPU, and then are moved to the devices they were saved
    # from, so we don't need to manually transfer the model to the GPU
    model = torch.jit.load(saved_model)
    model = model.to(device)
    input_tensor_gpu = input_tensor.to(torch.device(device))
    output_gpu = model.forward(input_tensor_gpu)
    output = output_gpu.to(torch.device("cpu"))

    return output


if __name__ == "__main__":
    # TODO: Accept command line argument for device type
    device_type = "cuda"

    saved_model_file = f"saved_multigpu_model_{device_type}.pt"

    num_devices = 2

    for device_index in range(num_devices):
        device_to_run = f"{device_type}:{device_index}"

        batch_size_to_run = 1

        with torch.no_grad():
            result = deploy(saved_model_file, device_to_run, batch_size_to_run)

        print(f"Output on device {device_to_run}: {result}")
