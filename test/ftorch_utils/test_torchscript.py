"""Unit tests for torchscript module."""

import os
import re

import pytest
import torch

from ftorch_utils.torchscript import (
    load_pytorch,
    load_torchscript,
    script_to_torchscript,
    trace_to_torchscript,
)


class SimpleNet(torch.nn.Module):
    """Everyone's favourite 'Hello, World!' net but with custom weights."""

    def __init__(self, weights=None):
        super().__init__()
        self._fwd_seq = torch.nn.Sequential(torch.nn.Linear(5, 5, bias=False))
        weight = 2.0 if weights is None else float(weights)
        with torch.inference_mode():
            self._fwd_seq[0].weight = torch.nn.Parameter(weight * torch.eye(5))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self._fwd_seq(batch)


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


def test_script_to_torchscript_creates_file(filename):
    """Check that `script_to_torchscript` successfully creates a file."""
    model = SimpleNet()
    script_to_torchscript(model, filename)
    assert os.path.exists(filename)


def test_trace_to_torchscript_creates_file(filename):
    """Check that `trace_to_torchscript` successfully creates a file."""
    model = SimpleNet()
    model.eval()
    tensor = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    trace_to_torchscript(model, tensor, filename)
    assert os.path.exists(filename)


def test_load_pytorch_from_file_default_weights(filename):
    """Check that `load_pytorch` is able to load a file in PyTorch pickled format."""
    torch.save(SimpleNet().state_dict(), filename)
    model = load_pytorch(
        "SimpleNet",
        model_definition_file=os.path.abspath(__file__),
        saved_model_file=filename,
    )
    assert isinstance(model, torch.nn.Module)
    weight_tensor = list(model.state_dict().values())[0]
    assert torch.allclose(weight_tensor, 2.0 * torch.eye(5))


def test_load_pytorch_from_file_custom_weights(filename):
    """Check that `load_pytorch` is able to load a file with custom weights."""
    torch.save(SimpleNet().state_dict(), filename)
    model = load_pytorch(
        "SimpleNet",
        model_definition_file=os.path.abspath(__file__),
        saved_model_file=filename,
        model_weights="3.0",
    )
    assert isinstance(model, torch.nn.Module)
    weight_tensor = list(model.state_dict().values())[0]
    assert torch.allclose(weight_tensor, 3.0 * torch.eye(5))


def test_load_pytorch_pretrained_default_weights():
    """Check that `load_pytorch` is able to load a pre-trained model."""
    model = load_pytorch("resnet18")
    assert isinstance(model, torch.nn.Module)


def test_load_pytorch_pretrained_custom_weights():
    """Check that `load_pytorch` is able to load a pre-trained model with custom weights."""
    model = load_pytorch("resnet18", model_weights="IMAGENET1K_V1")
    assert isinstance(model, torch.nn.Module)


def test_load_pytorch_missing_model_definition_file(filename):
    """Check that `load_pytorch` raises the expected error if configured incorrectly."""
    expected = (
        "Model definition file and input model file must either both be provided (to"
        " load a model from file) or both be skipped (to load a pre-trained model)."
    )
    with pytest.raises(ValueError, match=re.escape(expected)):
        load_pytorch("SimpleNet", saved_model_file=filename)


def test_load_pytorch_missing_saved_model_file(filename):
    """Check that `load_pytorch` raises the expected error if configured incorrectly."""
    expected = (
        "Model definition file and input model file must either both be provided (to"
        " load a model from file) or both be skipped (to load a pre-trained model)."
    )
    with pytest.raises(ValueError, match=re.escape(expected)):
        load_pytorch("SimpleNet", model_definition_file=os.path.abspath(__file__))


def test_load_torchscript(filename):
    """Check that `load_torchscript` is able to load a file in TorchScript format."""
    test_script_to_torchscript_creates_file(filename)
    model = load_torchscript(filename)
    assert isinstance(model, torch.jit._script.RecursiveScriptModule)
