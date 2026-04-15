"""Unit tests for validation module."""

import os
import warnings

import pytest
import torch

from ftorch_utils.validation import (
    validate_device_types,
    validate_input_model_file,
    validate_input_tensor_file,
    validate_output_model_file,
    validate_output_tensors,
)


@pytest.fixture
def filename():
    """Ensure a temporary PyTorch file is deleted during both setup and teardown."""
    # Setup
    file_name = "tmp.pt"
    if os.path.exists(file_name):
        os.remove(file_name)
    yield file_name

    # Teardown
    if os.path.exists(file_name):
        os.remove(file_name)
    assert not os.path.exists(file_name)


class TestValidateInputModel:
    """Tests for validate_input_model_file."""

    def test_invalid_extension(self):
        """Check that an error is raised for invalid input model file extension."""
        expected = "PyTorch input model file 'model.py' has extension .py but .pt was expected."
        with pytest.raises(ValueError, match=expected):
            validate_input_model_file("model.py")

    def test_model_does_not_exist(self, filename):
        """Check that an error is raised if an input model file doesn't exist."""
        expected = f"PyTorch input model file '{filename}' cannot be found."
        with pytest.raises(FileNotFoundError, match=expected):
            validate_input_model_file(filename)


class TestValidateInputTensor:
    """Tests for validate_input_tensor_file."""

    def test_invalid_extension(self):
        """Check that an error is raised for invalid input tensor file extension."""
        expected = "PyTorch input tensor file 'tensor.py' has extension .py but .pt was expected."
        with pytest.raises(ValueError, match=expected):
            validate_input_tensor_file("tensor.py")

    def test_tensor_does_not_exist(self, filename):
        """Check that an error is raised if an input tensor file doesn't exist."""
        expected = f"PyTorch input tensor file '{filename}' cannot be found."
        with pytest.raises(FileNotFoundError, match=expected):
            validate_input_tensor_file(filename)


class TestValidateOutputModel:
    """Tests for validate_output_model_file."""

    def test_invalid_extension(self):
        """Check that an error is raised for invalid output model file extension."""
        expected = "TorchScript output model file 'output.py' has extension .py but .pt was expected."
        with pytest.raises(ValueError, match=expected):
            validate_output_model_file("output.py", "input.pt")

    def test_input_model_matching_file(self):
        """Check that an error is raised if the input and output file names match."""
        expected = (
            "TorchScript output model file name 'input.pt' coincides with PyTorch input"
            " model file name 'input.pt'. It would be overwritten."
        )
        with pytest.raises(ValueError, match=expected):
            validate_output_model_file("input.pt", "input.pt")

    def test_file_exists_warning(self, filename):
        """Check that a warning is raised if the output file name already exists."""
        # Create a fake model file
        with open(filename, "w+") as f:
            f.write("TEST FILE")

        # Check that the expected warning is raised
        expected = (
            f"A file already exists with TorchScript output model file name '{filename}'."
            " It will be overwritten."
        )
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            validate_output_model_file(filename, "input.pt")
            assert len(warning_list) == 1
            assert issubclass(warning_list[0].category, UserWarning)
            assert str(warning_list[0].message) == expected


class TestValidateOutputTensors:
    """Tests for validate_output_tensors."""

    def test_validate_output_tensors_matching(self):
        """Check that matching tensors are accepted."""
        try:
            validate_output_tensors(
                torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])
            )
        except RuntimeError:
            pytest.fail("validate_output_tensors raised RuntimeError unexpectedly!")

    def test_mismatching(self):
        """Check that mismatching tensors are rejected."""
        expected = (
            "Saved Torchscript model is not performing as expected.\n"
            "Consider using scripting if you used tracing, or investigate further."
        )
        with pytest.raises(RuntimeError, match=expected):
            validate_output_tensors(
                torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.1, 3.0])
            )

    def test_mismatching_number(self):
        """Check that mismatching numbers of tensors are rejected."""
        expected = "argument 2 is shorter than argument 1"
        with pytest.raises(ValueError, match=expected):
            validate_output_tensors(
                torch.tensor([1.0]), (torch.tensor([1.0]), torch.tensor([1.0]))
            )

    def test_tuple_matching(self):
        """Check that matching tuples of tensors are accepted."""
        try:
            validate_output_tensors(
                (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
                (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
            )
        except RuntimeError:
            pytest.fail("validate_output_tensors raised RuntimeError unexpectedly!")

    def test_tuple_mismatching(self):
        """Check that mismatching tuples of tensors are rejected."""
        expected = (
            "Saved Torchscript model is not performing as expected.\n"
            "Consider using scripting if you used tracing, or investigate further."
        )
        with pytest.raises(RuntimeError, match=expected):
            validate_output_tensors(
                (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
                (torch.tensor([1.0, 2.0]), torch.tensor([3.1, 4.0])),
            )


class TestValidateDeviceTypes:
    """Tests for validate_device_types."""

    def test_validate_device_types_matching_cpu(self):
        """Check that matching CPU device types are accepted."""
        try:
            validate_device_types(
                torch.nn.Linear(2, 2).to("cpu"), torch.tensor([1.0, 2.0, 3.0]).to("cpu")
            )
        except RuntimeError:
            pytest.fail("validate_device_types raised RuntimeError unexpectedly!")

    def test_validate_device_types_matching_cuda(self):
        """Check that matching CUDA device types are accepted."""
        if torch.cuda.device_count() == 0:
            pytest.skip("No CUDA devices available, skipping test.")
        try:
            validate_device_types(
                torch.nn.Linear(2, 2).to("cuda"),
                torch.tensor([1.0, 2.0, 3.0]).to("cuda"),
            )
        except RuntimeError:
            pytest.fail("validate_device_types raised RuntimeError unexpectedly!")

    def test_validate_device_types_mismatching_cpu_cuda(self):
        """Check that mismatching device types raise an error."""
        if torch.cuda.device_count() == 0:
            pytest.skip("No CUDA devices available, skipping test.")
        expected = (
            "The model is on a different device from input tensor 0 ('cpu' vs."
            " 'cuda:0'). Ensure they are on the same device and try again."
        )
        with pytest.raises(RuntimeError, match=expected):
            validate_device_types(
                torch.nn.Linear(2, 2).to("cpu"),
                torch.tensor([1.0, 2.0, 3.0]).to("cuda"),
            )

    def test_validate_model_device_types_mismatching_cpu_cuda(self):
        """Check that mismatching device types in model parameters raise an error."""
        if torch.cuda.device_count() == 0:
            pytest.skip("No CUDA devices available, skipping test.")
        expected = (
            "The model has parameters on different devices ('cpu' vs. 'cuda:0')."
            " Ensure all model parameters are on the same device and try again."
        )
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2).to("cpu"),
            torch.nn.Linear(2, 2).to("cuda"),
        )
        with pytest.raises(RuntimeError, match=expected):
            validate_device_types(model, None)
