"""Unit tests for validation module."""

import pytest

from ftorch_utils.validation import (
    validate_input_model_file,
    validate_input_tensor_file,
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
