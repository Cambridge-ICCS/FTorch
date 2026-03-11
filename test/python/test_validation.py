"""Unit tests for validation module."""

import os
import warnings

import pytest

from ftorch_utils.validation import (
    validate_input_model_file,
    validate_input_tensor_file,
    validate_output_model_file,
)


def test_input_model_extension():
    expected = (
        "PyTorch input model file 'model.py' has extension .py but .pt was expected."
    )
    with pytest.raises(ValueError, match=expected):
        validate_input_model_file("model.py")


def test_input_model_exists():
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
    with open("tmp.pt", "w+") as f:
        f.write("TEST FILE")
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
    os.remove("tmp.pt")
