"""
Demonstrate batching and higher-dimensional inference with BatchingNet.

This script shows how a model trained on 1D vectors can be used for batched and
higher-dimensional inference in PyTorch. It includes examples for simple batches,
higher-dimensional data, and error handling for shape mismatches.
"""

import torch
from batchingnet import BatchingNet


def run_unbatched(model: torch.nn.Module) -> None:
    """
    Run inference on a single (unbatched) input vector.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for inference.
    """
    print("\n--- Unbatched Inference ---")
    input_tensor = torch.ones(5)  # shape: [5]
    print(f"Input shape: {input_tensor.shape}")
    print(f"Input: {input_tensor}")
    output = model(input_tensor)
    print(f"Output: {output}")


def run_simple_batch(model: torch.nn.Module) -> None:
    """
    Run inference on a simple batch of 1D vectors.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for inference.
    """
    print("\n--- Simple Batch Inference ---")
    # Create a batch of two input vectors, each of length 5
    input_tensor = torch.stack(
        [
            torch.ones(5),  # [1, 1, 1, 1, 1]
            2 * torch.ones(5),  # [2, 2, 2, 2, 2]
        ]
    )  # shape: [2, 5]
    print(f"Input shape: {input_tensor.shape}")
    print(f"Input:\n{input_tensor}")
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")


def run_higher_dim_batch(model: torch.nn.Module) -> None:
    """
    Run inference on higher-dimensional input e.g., [batch, time, features].

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for inference.
    """
    print("\n--- Higher-Dimensional Batch Inference ---")
    # Create a tensor with shape [batch=2, time=3, features=5] with fixed values for reproducibility
    input_tensor = torch.stack(
        [
            torch.stack(
                [
                    torch.ones(5),  # [1, 1, 1, 1, 1]
                    2 * torch.ones(5),  # [2, 2, 2, 2, 2]
                    3 * torch.ones(5),  # [3, 3, 3, 3, 3]
                ]
            ),
            torch.stack(
                [
                    10 * torch.ones(5),  # [10, 10, 10, 10, 10]
                    20 * torch.ones(5),  # [20, 20, 20, 20, 20]
                    30 * torch.ones(5),  # [30, 30, 30, 30, 30]
                ]
            ),
        ]
    )  # shape: [2, 3, 5]
    print(f"Input shape: {input_tensor.shape}")
    print(f"Input:\n{input_tensor}")
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")


def run_shape_error_case(model: torch.nn.Module) -> None:
    """
    Demonstrate error handling for input with wrong feature dimension.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for inference.
    """
    print("\n--- Error Case: Wrong Feature Dimension ---")
    # Input tensor with wrong dimension order
    # 5, the feature dimension expected by the net, should be last,
    # preceded by any batching dimensions.
    input_tensor = torch.ones(5, 2)
    print(f"Input shape: {input_tensor.shape}")
    try:
        output = model(input_tensor)
        print(f"Output:\n{output}")
    except RuntimeError as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    # Set up the BatchingNet model
    model = BatchingNet()
    model.eval()

    # Run all batching demos
    run_unbatched(model)
    run_simple_batch(model)
    run_higher_dim_batch(model)
    run_shape_error_case(model)
