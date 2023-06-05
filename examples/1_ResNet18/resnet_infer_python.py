"""Load ResNet-18 saved to TorchScript and run inference with ones."""

import torch


def deploy(saved_model, device, batch_size=1):
    """
    Load TorchScript ResNet-18 and run inference with Tensor of ones.

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

    input_tensor = torch.ones(batch_size, 3, 224, 224)

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


if __name__ == "__main__":

    saved_model_file = "saved_resnet18_model_cpu.pt"

    device_to_run = "cpu"
    # device = "cuda"

    batch_size_to_run = 1

    result = deploy(saved_model_file, device_to_run, batch_size_to_run)

    print(result[:, 0:5])
