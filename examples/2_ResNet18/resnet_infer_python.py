"""Load ResNet-18 saved to TorchScript and run inference with an example image."""

import numpy as np
import torch


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
    transposed_shape = [224, 224, 3, 1]
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


def print_top_results(output: torch.Tensor) -> None:
    """Prints top 5 results

    Parameters
    ----------
    output: torch.Tensor
        Output from ResNet-18.
    """
    #  Run a softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Read ImageNet labels from text file
    cats_filename = "data/categories.txt"
    categories = np.genfromtxt(cats_filename, dtype=str, delimiter="\n")

    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    print("\nTop 5 results:\n")
    for i in range(top5_prob.size(0)):
        cat_id = top5_catid[i]
        print(
            f"{categories[cat_id]} (id={cat_id}): probability = {top5_prob[i].item()}"
        )


if __name__ == "__main__":
    saved_model_file = "saved_resnet18_model_cpu.pt"

    device_to_run = "cpu"
    # device_to_run = "cuda"

    batch_size_to_run = 1

    output = deploy(saved_model_file, device_to_run, batch_size_to_run)
    print_top_results(output)
