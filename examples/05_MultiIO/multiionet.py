"""Example using the reusable MultiIONet model from ftorch-utils."""

import torch

from ftorch_utils.models import MultiIONet

if __name__ == "__main__":
    import argparse

    # Parse user input
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device_type",
        help="Device type to run the inference on",
        type=str,
        choices=["cpu", "cuda", "hip", "xpu", "mps"],
        default="cpu",
    )
    parsed_args = parser.parse_args()
    device_type = parsed_args.device_type

    # Construct an instance of the MultiIONet model on the specified device
    model = MultiIONet().to(device_type)
    model.eval()

    # Save the model in PyTorch format
    torch.save(model.state_dict(), f"pytorch_multiio_model_{device_type}.pt")

    # Create arbitrary input tensors and save them in PyTorch format
    input_tensors = (
        torch.Tensor([0.0, 1.0, 2.0, 3.0]).to(device_type),
        torch.Tensor([-0.0, -1.0, -2.0, -3.0]).to(device_type),
    )
    torch.save(input_tensors, f"pytorch_multiio_input_tensor_{device_type}.pt")

    # Propagate the input tensors through the model
    with torch.inference_mode():
        output_tensors = model(*input_tensors)
    print(f"Model output: {output_tensors}")

    # Perform a basic check of the model output
    for output_i, input_i, scale_factor in zip(output_tensors, input_tensors, (2, 3)):
        if not torch.allclose(output_i, scale_factor * input_i):
            result_error = (
                f"result:\n{output_i}\ndoes not match expected value:\n"
                f"{scale_factor * input_i}"
            )
            raise ValueError(result_error)
