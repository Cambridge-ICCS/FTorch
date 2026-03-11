#!/bin/env python
"""Convert a PyTorch model file to TorchScript format."""

import argparse
import os
from warnings import warn

import torch

from ftorch_utils.torchscript import (
    load_pytorch,
    load_torchscript,
    script_to_torchscript,
    trace_to_torchscript,
)
from ftorch_utils.validation import (
    validate_input_model_file,
    validate_input_tensor_file,
    validate_output_model_file,
    validate_output_tensors,
)


def parse_user_input():
    """Retrieve command line options.

    Returns
    -------
    Namespace
        namespace containing known input arguments
    """
    parser = argparse.ArgumentParser(
        prog="pt2ts.py",
        description="Convert a PyTorch model file to TorchScript format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_definition_file",
        help="Filename for the definition of the PyTorch model, including path",
        type=str,
    )
    parser.add_argument(
        "model_name",
        help="Name of the PyTorch model",
        type=str,
    )
    # TODO: Accept pre-trained model name (needed for ResNet)
    #       Perhaps make model_definition_file optional? Need to allow weights
    #       specification, too
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
        "--input_tensor_file",
        help=(
            "Filename for the tensor(s) saved in PyTorch format, including path (only"
            " required if running with --trace or --test). The tensor is used to"
            " determine the dimensionality of the inputs so its values are ignored."
        ),
        type=str,
    )
    parser.add_argument(
        "--trace",
        help=(
            "Apply tracing rather than scripting.\n\n"
            "If used then --input_tensor_file must also be provided."
        ),
        action="store_true",
    )
    parser.add_argument(
        "--test",
        help=(
            "Run basic testing to check everything is working as expected.\n\n"
            "If used then --input_tensor_file must also be provided."
        ),
        action="store_true",
    )
    parsed_args = parser.parse_args()
    if parsed_args.input_tensor_file is None and parsed_args.trace:
        value_error = "An input tensor must be provided to use --trace."
        raise ValueError(value_error)
    if parsed_args.input_tensor_file is None and parsed_args.test:
        value_error = "An input tensor must be provided to use --test."
        raise ValueError(value_error)
    return parsed_args


def main_cli():
    """Driver for command line interface to the `pt2ts` script.

    This function is required for the version that gets installed into the Python
    environment's `bin` subdirectory.
    """
    parsed_args = parse_user_input()
    model_definition_file = parsed_args.model_definition_file
    model_name = parsed_args.model_name
    input_model_file = parsed_args.input_model_file
    output_model_file = parsed_args.output_model_file
    trace = parsed_args.trace
    test = parsed_args.test
    input_tensor_file = parsed_args.input_tensor_file

    validate_input_model_file(input_model_file)
    if output_model_file is None:
        output_model_file = input_model_file
    validate_output_model_file(output_model_file, input_model_file)

    # Load the input PyTorch model
    model = load_pytorch(model_definition_file, model_name, input_model_file)

    if test or trace:
        validate_input_tensor_file(input_tensor_file)

        # Load the input tensor
        input_tensors = torch.load(input_tensor_file)
        if not isinstance(input_tensors, tuple):
            input_tensors = (input_tensors,)

    if test:
        # Propagate the input tensor through the model
        # If something isn't working This will generate an error
        pt_model_outputs = model(*input_tensors)

    # Apply scripting or tracing as requested, writing out to file
    if trace:
        trace_to_torchscript(model, input_tensors, output_model_file)
    else:
        script_to_torchscript(model, output_model_file)
    print(f"Saved model to TorchScript in '{output_model_file}'.")

    if test:
        # Check that the model file is created
        if not os.path.exists(output_model_file):
            torchscript_file_error = (
                f"Saved TorchScript file '{output_model_file}' cannot be found."
            )
            raise FileNotFoundError(torchscript_file_error)

        # Load the TorchScript model and propagate the same input tensor
        ts_model = load_torchscript(output_model_file)
        ts_model_outputs = ts_model(*input_tensors)
        validate_output_tensors(pt_model_outputs, ts_model_outputs)
        print("Saved TorchScript model working as expected in a basic test.")
        print("Users should perform further validation as appropriate.")


if __name__ == "__main__":
    main()
