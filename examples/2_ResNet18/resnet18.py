"""Load and run pretrained ResNet-18 from TorchVision."""

import numpy as np
from PIL import Image
import torch
import torchvision
import urllib


# Initialize everything
def initialize():
    """
    Download pre-trained ResNet-18 model and prepare for inference.

    Returns
    -------
    model : torch.nn.Module
    """

    # Load a pre-trained PyTorch model
    print("Loading pre-trained ResNet-18 model...", end="")
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    print("done.")

    # Switch-off some specific layers/parts of the model that behave
    # differently during training and inference
    model.eval()

    return model


def run_model(model):
    """
    Run the pre-trained ResNet-18 with an example image of a dog.

    Parameters
    ----------
    model : torch.nn.Module
    """

    print("Downloading image...", end="")
    url, filename = (
        "https://github.com/pytorch/hub/raw/e55b003/images/dog.jpg",
        "dog.jpg",
    )
    urllib.request.urlretrieve(url, filename)
    print("done.")

    # Transform image into the  form expected by the model
    input_image = Image.open(filename)
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
    dtype = np.float32
    np_input = np.array(input_batch.numpy().flatten(), dtype=dtype)
    np_input.tofile("image_tensor.dat")

    # Check saved correctly
    tensor_shape = input_batch.shape
    np_data = np.fromfile("image_tensor.dat", dtype=dtype).reshape(tensor_shape)

    assert np.array_equal(np_data, input_batch.numpy()) is True
    print("done.")

    print("Running ResNet-18 model for input...", end="")
    with torch.no_grad():
        output = model(input_batch)
    print("done.")

    #  Run a softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Download ImageNet labels
    url, filename = (
        "https://raw.githubusercontent.com/pytorch/hub/e55b003/imagenet_classes.txt",
        "imagenet_classes.txt",
    )
    data = urllib.request.urlopen(url)
    for _ in data:
        categories = [s.strip() for s in data]

    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    print("\nTop 5 results:\n")
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())


if __name__ == "__main__":
    rn_model = initialize()
    run_model(rn_model)
