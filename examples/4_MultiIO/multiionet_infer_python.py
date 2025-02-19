"""Load saved MultIONet to TorchScript and run inference example."""

import os

import torch


def deploy(saved_model: str, device: str, batch_size: int = 1) -> torch.Tensor:
    """
    Load TorchScript MultiIONet and run inference with example Tensors.

    Parameters
    ----------
    saved_model : str
        location of MultiIONet model saved to Torchscript
    device : str
        Torch device to run model on, 'cpu' or 'cuda'
    batch_size : int
        batch size to run (default 1)

    Returns
    -------
    outputs : tuple
        result of running inference on model with example Tensor inputs
    """
    input_tensors = (
        torch.tensor([0.0, 1.0, 2.0, 3.0]).repeat(batch_size, 1),
        torch.tensor([0.0, -1.0, -2.0, -3.0]).repeat(batch_size, 1),
    )

    if device == "cpu":
        # Load saved TorchScript model
        model = torch.jit.load(saved_model)
        # Inference
        outputs = model.forward(*input_tensors)

    elif device == "cuda":
        # All previously saved modules, no matter their device, are first
        # loaded onto CPU, and then are moved to the devices they were saved
        # from, so we don't need to manually transfer the model to the GPU
        model = torch.jit.load(saved_model)
        input_tensors_gpu = input_tensors.to(torch.device("cuda"))
        outputs_gpu = model.forward(*input_tensors_gpu)
        outputs = outputs_gpu.to(torch.device("cpu"))

    else:
        device_error = f"Device '{device}' not recognised."
        raise ValueError(device_error)

    return outputs


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
    saved_model_file = os.path.join(filepath, "saved_multiio_model_cpu.pt")

    device_to_run = "cpu"

    batch_size_to_run = 1

    with torch.no_grad():
        result = deploy(saved_model_file, device_to_run, batch_size_to_run)

    print(result)

    expected_tensors = (
        torch.Tensor([0.0, 2.0, 4.0, 6.0]),
        torch.Tensor([-0.0, -3.0, -6.0, -9.0]),
    )

    for got, expected in zip(result, expected_tensors):
        if not torch.allclose(got, expected):
            result_error = f"result:\n{got}\ndoes not match expected value:\n{expected}"
            raise ValueError(result_error)
