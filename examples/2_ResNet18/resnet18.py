import numpy as np
from PIL import Image
import torch
import torchvision
import argparse



# Initialize everything
def initialize(precision: torch.dtype) -> torch.nn.Module:
    """
    Download pre-trained ResNet-18 model and prepare for inference.

    Parameters
    ----------
    precision: torch.dtype
        Sets the working precision of the model.

    Returns
    -------
    model: torch.nn.Module
        Pretrained ResNet-18 model
    """

    # Set working precision
    torch.set_default_dtype(precision)

    # Load a pre-trained PyTorch model
    print("Loading pre-trained ResNet-18 model...", end="")
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    print("done.")

    # Switch-off some specific layers/parts of the model that behave
    # differently during training and inference
    model.eval()

    return model


def run_model(model: torch.nn.Module, precision: type, image_filename: str) -> None:
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
    np_input = np.array(
        input_batch.numpy().transpose().flatten(), dtype=precision
    )  # type: np.typing.NDArray

    # Save data as binary
    np_input.tofile("image_tensor.dat")

    # Load saved data to check it was saved correctly
    np_data = np.fromfile(
        "image_tensor.dat", dtype=precision
    )  # type: np.typing.NDArray

    # Reshape to original tensor shape
    tensor_shape = np.array(input_batch.numpy()).transpose().shape
    np_data = np_data.reshape(tensor_shape)
    np_data = np_data.transpose()
    assert np.array_equal(np_data, input_batch.numpy()) is True
    print("done.")

    print("Running ResNet-18 model for input...", end="")
    with torch.no_grad():
        output = model(input_batch)
    print("done.")

    print(output)


 
if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="ResNet-18 Image Classification")
    parser.add_argument("image_path", type=str, help="Path to the input image file")
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp64"],
        default="fp32",
        help="Working precision (default: fp32)",
    )
    parser.add_argument(
        "--model_type",
        choices=["resnet18"],
        default="resnet18",
        help="Model type (default: resnet18)",
    )
    args = parser.parse_args()

    # Map command-line precision argument to NumPy and Torch data types
    if args.precision == "fp32":
        np_precision = np.float32
        torch_precision = torch.float32
    elif args.precision == "fp64":
        np_precision = np.float64
        torch_precision = torch.float64
    else:
        raise ValueError("Precision must be 'fp32' or 'fp64'")

    rn_model = initialize(torch_precision)
    run_model(rn_model, np_precision, args.image_path)
