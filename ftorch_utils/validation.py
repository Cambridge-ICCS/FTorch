"""Module containing validation functions for pt2ts input and output files."""

import os
import sys

__all__ = [
    "validate_input_model_file",
    "validate_output_model_file",
    "validate_input_tensor_file",
]


def validate_input_model_file(input_model_file):
    """Check the input model file exists and has the correct extension.

    Parameters
    ----------
    input_model_file : str
        Name of the input model file
    """
    input_model_root, input_model_ext = os.path.splitext(input_model_file)
    if input_model_ext != ".pt":
        value_error = (
            f"PyTorch input model file '{input_model_file}' has extension"
            f" {input_model_ext} but .pt was expected."
        )
        raise ValueError(value_error)
    if not os.path.exists(input_model_file):
        input_file_error = (
            f"PyTorch input model file '{input_model_file}' cannot be found."
        )
        raise FileNotFoundError(input_file_error)


def validate_output_model_file(output_model_file, input_model_file):
    """Check the output model file has the correct file extension.

    Also raise a warning if it would overwrite an existing file.

    Parameters
    ----------
    output_model_file : str
        Name of the output model file
    input_model_file : str
        Name of the input model file
    """
    if output_model_file is None:
        output_model_file = input_model_file
    _, output_model_ext = os.path.splitext(output_model_file)
    if output_model_ext != ".pt":
        value_error = (
            f"TorchScript output file name '{output_model_file}' has extension"
            f" {output_model_ext} but .pt was expected."
        )
        raise ValueError(value_error)
    if input_model_file == output_model_file:
        warning = (
            f"Output TorchScript file name '{output_model_file}' coincides with input"
            f" PyTorch file name '{input_model_file}'. It will be overwritten."
        )
        warn(warning, stacklevel=2)
    elif os.path.exists(output_model_file):
        warning = (
            "A file already exists with output TorchScript file name"
            f" '{output_model_file}'. It will be overwritten."
        )
        warn(warning, stacklevel=2)


def validate_input_tensor_file(input_tensor_file):
    """Check the input tensor file exists and has the correct extension.

    Parameters
    ----------
    input_tensor_file : str
        Name of the input model file
    """
    _, input_tensor_ext = os.path.splitext(input_tensor_file)
    if input_tensor_ext != ".pt":
        value_error = (
            f"PyTorch input tensor file '{input_tensor_file}' has extension"
            f" {input_tensor_ext} but .pt was expected."
        )
        raise ValueError(value_error)
    if not os.path.exists(input_tensor_file):
        input_file_error = (
            f"PyTorch input tensor file '{input_tensor_file}' cannot be found."
        )
        raise FileNotFoundError(input_file_error)
