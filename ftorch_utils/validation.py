"""Module containing validation functions for pt2ts input and output files."""

import os
import sys
import warnings

import torch

__all__ = [
    "validate_file_exists",
    "validate_input_model_file",
    "validate_input_tensor_file",
    "validate_output_model_file",
    "validate_output_tensors",
]


def validate_file_exists(filename, description):
    """Check a file exists.

    Parameters
    ----------
    filename : str
        Name of the file
    description : str
        Brief description of the file contents
    """
    if not os.path.exists(filename):
        input_file_error = f"{description} file '{filename}' cannot be found."
        raise FileNotFoundError(input_file_error)


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
    validate_file_exists(input_model_file, "PyTorch input model")


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
    validate_file_exists(input_tensor_file, "PyTorch input tensor")


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
    _, output_model_ext = os.path.splitext(output_model_file)
    if output_model_ext != ".pt":
        value_error = (
            f"TorchScript output model file '{output_model_file}' has extension"
            f" {output_model_ext} but .pt was expected."
        )
        raise ValueError(value_error)
    if input_model_file == output_model_file:
        warning = (
            f"TorchScript output model file name '{output_model_file}' coincides with"
            " input PyTorch model file name. It will be overwritten."
        )
        warnings.warn(warning, stacklevel=2)
    elif os.path.exists(output_model_file):
        warning = (
            "A file already exists with TorchScript output model file name"
            f" '{output_model_file}'. It will be overwritten."
        )
        warnings.warn(warning, stacklevel=2)


def validate_output_tensors(expected_tensor, result_tensor):
    """Check the output tensors from the TorchScript model match the expected tensors.

    Parameters
    ----------
    expected_tensor : torch.Tensor or tuple of torch.Tensor
        The expected output tensor(s) from the TorchScript model, as obtained by running
        the original PyTorch model on the input tensors.

    result_tensor : torch.Tensor or tuple of torch.Tensor
        The output tensor(s) from the TorchScript model, as obtained by running the
        TorchScript model on the input tensors.
    """
    if not isinstance(result_tensor, tuple):
        result_tensor = (result_tensor,)
    if not isinstance(expected_tensor, tuple):
        expected_tensor = (expected_tensor,)
    for result, expected in zip(result_tensor, expected_tensor, strict=True):
        if not torch.all(result.eq(expected)):
            model_error = (
                "Saved Torchscript model is not performing as expected.\n"
                "Consider using scripting if you used tracing, or investigate further."
            )
            raise RuntimeError(model_error)
