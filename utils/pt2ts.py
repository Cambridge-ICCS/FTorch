#!/bin/env python
"""Convert a PyTorch model file to TorchScript format."""

import torch


def script_to_torchscript(model: torch.nn.Module, filename: str) -> None:
    """
    Save PyTorch model to TorchScript using scripting.

    Parameters
    ----------
    model : torch.NN.Module
        a PyTorch model
    filename : str
        name of file to save to
    """
    # FIXME: Adopt torch.jit.optimize_for_inference() once
    # https://github.com/pytorch/pytorch/issues/81085 is resolved
    scripted_model = torch.jit.script(model)
    # print(scripted_model.code)
    scripted_model.save(filename)


def trace_to_torchscript(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    filename: str,
) -> None:
    """
    Save PyTorch model to TorchScript using tracing.

    Parameters
    ----------
    model : torch.NN.Module
        a PyTorch model
    input_tensor : torch.Tensor
        appropriate size Tensor to act as input to model
    filename : str
        name of file to save to
    """
    # FIXME: Adopt torch.jit.optimize_for_inference() once
    # https://github.com/pytorch/pytorch/issues/81085 is resolved
    traced_model = torch.jit.trace(model, input_tensor)
    # traced_model.save(filename)
    frozen_model = torch.jit.freeze(traced_model)
    ## print(frozen_model.graph)
    ## print(frozen_model.code)
    frozen_model.save(filename)


def load_torchscript(filename: str) -> torch.nn.Module:
    """
    Load a TorchScript model from file.

    Parameters
    ----------
    filename : str
        name of file containing TorchScript model
    """
    return torch.jit.load(filename)


if __name__ == "__main__":
    import argparse
    import os

    # Parse user input
    parser = argparse.ArgumentParser(
        prog="pt2ts.py",
        description="Convert a PyTorch model file to TorchScript format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_model_file",
        help="Filename for the model saved in PyTorch format, including path",
        type=str,
    )
    parser.add_argument(
        "--output_model_file",
        help="Filename for the model to be saved in TorchScript format, including path",
        type=str,
    )
    parser.add_argument(
        "--trace",
        help="Apply tracing rather than scripting",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--input_tensor_file",
        help=(
            "Filename for the tensor saved in PyTorch format, including path (only"
            " required if running with --trace). The tensor is used to determine the"
            " dimensionality of the inputs so its values are ignored."
        ),
        type=str,
    )
    parsed_args = parser.parse_args()
    input_model_file = parsed_args.input_model_file
    output_model_file = parsed_args.output_model_file
    trace = parsed_args.trace
    input_tensor_file = parsed_args.input_tensor_file

    # Process input model file name
    input_model_root, input_model_ext = os.path.splitext(input_model_file)[1]
    if ext != ".pt":
        value_error = (
            f"PyTorch input file '{input_model_file}' has extension {input_model_ext}"
            " but .pt was expected."
        )
        raise ValueError(value_error)
    if not os.path.exists(input_model_file):
        input_file_error = f"PyTorch model file '{input_model_file}' cannot be found."
        raise FileNotFoundError(input_file_error)

    # Process output model file name
    if output_model_file is None:
        output_model_file = input_model_root + ".ts"
    _, output_model_ext = os.path.splitext(output_model_file)[1]
    if ext not in (".ts", ".pt"):
        value_error = (
            f"TorchScript output file name '{output_model_file}' has extension"
            f" {output_model_ext} but .ts (recommended) or .pt (legacy) was expected."
        )
        raise ValueError(value_error)
    if input_model_file == output_model_file:
        value_error = (
            f"Output TorchScript file name '{output_model_file}' coincides with input"
            f" PyTorch file name '{input_model_file}' and would overwrite it."
        )
        raise ValueError(value_error)
    if os.path.exists(output_model_file):
        value_error = (
            "A file already exists with output TorchScript file name"
            f" '{output_model_file}' and would be overwritten."
        )
        raise ValueError(value_error)

    # Load the input PyTorch model
    model = torch.load(input_model_file)

    # Apply scripting or tracing as requested, writing out to file
    if trace:
        # Process input tensor file name
        _, input_tensor_ext = os.path.splitext(input_model_file)[1]
        if input_tensor_ext != ".pt":
            value_error = (
                f"PyTorch input tensor file '{input_tensor_file}' has extension"
                f" {input_tensor_ext} but .pt was expected."
            )
            raise ValueError(value_error)
        if not os.path.exists(input_tensor_file):
            input_file_error = (
                f"PyTorch tensor file '{input_tensor_file}' cannot be found."
            )
            raise FileNotFoundError(input_file_error)

        # Load the input tensor
        input_tensor = torch.load(input_tensor_file)

        trace_to_torchscript(model, input_tensor, output_model_file)
    else:
        script_to_torchscript(model, output_model_file)
