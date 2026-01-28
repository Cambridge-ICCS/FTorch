"""
Demonstrate batching and higher-dimensional inference with SimpleNet.

This script shows how a model trained on 1D vectors can be used for batched and
higher-dimensional inference in PyTorch. It includes examples for simple batches,
higher-dimensional data, and error handling for shape mismatches.
"""

import torch
from simplenet import SimpleNet

def run_unbatched(model: torch.nn.Module) -> None:
    """
    Run inference on a single (unbatched) input vector.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for inference.
    """
    print("\n--- Unbatched Inference ---")
    input_tensor = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])  # shape: [5]
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
    input_tensor = torch.tensor([
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0, 9.0],
    ])  # shape: [2, 5]
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
    input_tensor = torch.tensor([
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0, 13.0, 14.0],
        ],
        [
            [100.0, 101.0, 102.0, 103.0, 104.0],
            [105.0, 106.0, 107.0, 108.0, 109.0],
            [110.0, 111.0, 112.0, 113.0, 114.0],
        ],
    ])  # shape: [2, 3, 5]
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
    # Input tensor with wrong last dimension (should be 5)
    input_tensor = torch.ones(5, 2)
    print(f"Input shape: {input_tensor.shape}")
    try:
        output = model(input_tensor)
        print(f"Output:\n{output}")
    except RuntimeError as e:
        print(f"Error occurred: {e}")

def main() -> None:
    """Run all batching demos."""
    model = SimpleNet()
    model.eval()
    run_unbatched(model)
    run_simple_batch(model)
    run_higher_dim_batch(model)
    run_shape_error_case(model)

if __name__ == "__main__":
    main()


