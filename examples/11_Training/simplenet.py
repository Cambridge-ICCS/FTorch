"""Example using the reusable SimpleNet model from ftorch-utils."""

import torch

from ftorch_utils.models import SimpleNet

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

    # Construct an instance of the SimpleNet model on the specified device
    model = SimpleNet().to(device_type)

    # Save the model in PyTorch format
    torch.save(model.state_dict(), f"pytorch_simplenet_model_{device_type}.pt")

    # Create an arbitrary input tensor and save it in PyTorch format
    input_tensor = torch.Tensor([0.0, 1.0, 2.0, 3.0, 4.0]).to(device_type)
    torch.save(input_tensor, f"pytorch_simplenet_input_tensor_{device_type}.pt")

    # Propagate the input tensor through the model
    with torch.inference_mode():
        output_tensor = model(input_tensor).to("cpu")
    print(f"Model output: {output_tensor}")

    # Perform a basic check of the model output
    if not torch.allclose(output_tensor, 2 * input_tensor):
        result_error = (
            f"result:\n{output_tensor}\ndoes not match expected value:\n"
            f"{2 * input_tensor}"
        )
        raise ValueError(result_error)
