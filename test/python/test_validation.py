"""Unit tests for validation module."""

import os
import warnings

import pytest
import torch

from ftorch_utils.validation import (
    validate_input_model_file,
    validate_input_tensor_file,
    validate_output_model_file,
    validate_output_tensors,
)


def test_input_model_extension():
    expected = (
        "PyTorch input model file 'model.py' has extension .py but .pt was expected."
    )
    with pytest.raises(ValueError, match=expected):
        validate_input_model_file("model.py")


def test_input_model_exists():
    if os.path.exists("model.pt"):
        os.remove("model.pt")
    expected = "PyTorch input model file 'model.pt' cannot be found."
    with pytest.raises(FileNotFoundError, match=expected):
        validate_input_model_file("model.pt")


def test_input_tensor_extension():
    expected = (
        "PyTorch input tensor file 'tensor.py' has extension .py but .pt was expected."
    )
    with pytest.raises(ValueError, match=expected):
        validate_input_tensor_file("tensor.py")


def test_input_tensor_exists():
    if os.path.exists("tensor.pt"):
        os.remove("tensor.pt")
    expected = "PyTorch input tensor file 'tensor.pt' cannot be found."
    with pytest.raises(FileNotFoundError, match=expected):
        validate_input_tensor_file("tensor.pt")


def test_output_model_extension():
    expected = "TorchScript output model file 'output.py' has extension .py but .pt was expected."
    with pytest.raises(ValueError, match=expected):
        validate_output_model_file("output.py", "input.pt")


def test_output_model_matching_file_warning():
    expected = (
        "TorchScript output model file name 'input.pt' coincides with input PyTorch"
        " model file name. It will be overwritten."
    )
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        validate_output_model_file("input.pt", "input.pt")
        assert len(warning_list) == 1
        assert issubclass(warning_list[0].category, UserWarning)
        assert str(warning_list[0].message) == expected


def test_output_model_file_exists_warning():
    if os.path.exists("tmp.pt"):
        os.remove("tmp.pt")

    # Create a fake model file
    with open("tmp.pt", "w+") as f:
        f.write("TEST FILE")

    # Check that the expected warning is raised
    expected = (
        "A file already exists with TorchScript output model file name 'tmp.pt'. It"
        " will be overwritten."
    )
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        validate_output_model_file("tmp.pt", "input.pt")
        assert len(warning_list) == 1
        assert issubclass(warning_list[0].category, UserWarning)
        assert str(warning_list[0].message) == expected

    # Remove the fake model file
    os.remove("tmp.pt")
    assert not os.path.exists("tmp.pt")


def test_validate_output_tensors_matching():
    try:
        validate_output_tensors(
            torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])
        )
    except RuntimeError:
        pytest.fail("validate_output_tensors raised RuntimeError unexpectedly!")


def test_validate_output_tensors_mismatching():
    expected = (
        "Saved Torchscript model is not performing as expected.\n"
        "Consider using scripting if you used tracing, or investigate further."
    )
    with pytest.raises(RuntimeError, match=expected):
        validate_output_tensors(
            torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.1, 3.0])
        )


def test_validate_output_tensors_tuple_matching():
    try:
        validate_output_tensors(
            (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
            (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
        )
    except RuntimeError:
        pytest.fail("validate_output_tensors raised RuntimeError unexpectedly!")


def test_validate_output_tensors_tuple_mismatching():
    with pytest.raises(
        RuntimeError, match="Saved Torchscript model is not performing as expected."
    ):
        validate_output_tensors(
            (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
            (torch.tensor([1.0, 2.0]), torch.tensor([3.1, 4.0])),
        )
