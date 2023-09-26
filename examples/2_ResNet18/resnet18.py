"""Load and run pretrained ResNet-18 from TorchVision."""

import numpy as np
from PIL import Image
import torch
import torchvision


# Initialize everything
def initialize(wp: type) -> torch.nn.Module:
    """
    Download pre-trained ResNet-18 model and prepare for inference.

    Parameters
    ----------
    wp: type
        Data type of input tensor.

    Returns
    -------
    model: torch.nn.Module
        Pretrained ResNet-18 model
    """

    # Set working precision
    torch.set_default_dtype(wp)

    # Load a pre-trained PyTorch model
    print("Loading pre-trained ResNet-18 model...", end="")
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    print("done.")

    # Switch-off some specific layers/parts of the model that behave
    # differently during training and inference
    model.eval()

    return model


def run_model(model: torch.nn.Module, wp: type):
    """
    Run the pre-trained ResNet-18 with an example image of a dog.

    Parameters
    ----------
    model: torch.nn.Module
        Pretrained model to run.
    wp: type
        Data type to save input tensor.
    """
    # Transform image into the form expected by the pre-trained model, using the mean
    # and standard deviation from the ImageNet dataset
    # See: https://pytorch.org/vision/0.8/models.html
    image_filename = "data/dog.jpg"
    input_image = Image.open(image_filename)
    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    print("Saving input batch...", end="")
    # Transpose input before saving so order consistent with Fortran
    np_input = np.array(input_batch.numpy().transpose().flatten(), dtype=wp)
    np_input.tofile("data/image_tensor.dat")

    # Check saved correctly
    tensor_shape = np.array(input_batch.numpy()).transpose().shape
    np_data = np.fromfile("data/image_tensor.dat", dtype=wp)
    np_data = np_data.reshape(tensor_shape)
    np_data = np_data.transpose()
    assert np.array_equal(np_data, input_batch.numpy()) is True
    print("done.")

    print("Running ResNet-18 model for input...", end="")
    with torch.no_grad():
        output = model(input_batch)
    print("done.")

    #  Run a softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Read ImageNet labels from text file
    cats_filename = "data/categories.txt"
    categories = np.genfromtxt(cats_filename, dtype=str, delimiter="\n")

    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    print("\nTop 5 results:\n")
    for i in range(top5_prob.size(0)):
        id = top5_catid[i]
        print(f"{categories[id]} (id={id}): probability = {top5_prob[i].item()}")


if __name__ == "__main__":
    wp = np.float32

    if wp == np.float32:
        wp_torch = torch.float32
    elif wp == np.float64:
        wp_torch = torch.float64
    else:
        raise ValueError("`wp` must be of type `np.float32` or `np.float64`")

    rn_model = initialize(wp_torch)
    run_model(rn_model, wp)