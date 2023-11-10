"""Load and run pretrained ResNet-18 from TorchVision."""

import torch
import torch.nn.functional as F
import torchvision


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
    model = torchvision.models.resnet18(pretrained=True)
    print("done.")

    # Switch-off some specific layers/parts of the model that behave
    # differently during training and inference
    model.eval()

    return model


def run_model(model):
    """
    Run the pre-trained ResNet-18 with dummy input of ones.

    Parameters
    ----------
    model : torch.nn.Module
    """
    print("Running ResNet-18 model for ones...", end="")
    dummy_input = torch.ones(1, 3, 224, 224)
    output = model(dummy_input)
    top5 = F.softmax(output, dim=1).topk(5).indices
    print("done.")

    print(f"Top 5 results:\n  {top5}")


if __name__ == "__main__":
    rn_model = initialize()
    run_model(rn_model)
