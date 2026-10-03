"""
Example using the reusable BatchingNet model from ftorch-utils.
"""

import torch

from ftorch_utils.models import BatchingNet


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

    # Construct an instance of the BatchingNet model on the specified device
    model = BatchingNet().to(device_type)
    model.eval()

    # Save the model in PyTorch format
    torch.save(model.state_dict(), f"pytorch_batchingnet_model_{device_type}.pt")

    # Create an arbitrary input tensor and save it in PyTorch format
    input_tensor = torch.ones(5).to(device_type)
    torch.save(input_tensor, f"pytorch_batchingnet_input_tensor_{device_type}.pt")

    # Propagate the input tensor through the model
    with torch.inference_mode():
        output_tensor = model(input_tensor)
    print(f"Model output: {output_tensor}")

    # Perform a basic check of the model output
    if not torch.allclose(
        output_tensor,
        torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
    ):
        result_error = (
            f"result:\n{output_tensor}\ndoes not match expected value:\n"
            f"{2 * input_tensor}"
        )
        raise ValueError(result_error)