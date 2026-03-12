"""Unit tests for torchscript module."""

import os

import pytest
import torch

from ftorch_utils.torchscript import (
    load_pytorch,
    load_torchscript,
    script_to_torchscript,
    trace_to_torchscript,
)


class SimpleNet(torch.nn.Module):
    """Everyone's favourite 'Hello, World!' net."""

    def __init__(self):
        super().__init__()
        self._fwd_seq = torch.nn.Sequential(torch.nn.Linear(5, 5, bias=False))
        with torch.inference_mode():
            self._fwd_seq[0].weight = torch.nn.Parameter(2.0 * torch.eye(5))

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


def test_load_pytorch(filename):
    """Check that `load_pytorch` is able to load a file in PyTorch pickled format."""
    torch.save(SimpleNet().state_dict(), filename)
    model = load_pytorch(os.path.abspath(__file__), "SimpleNet", filename)
    assert isinstance(model, torch.nn.Module)


def test_load_torchscript(filename):
    """Check that `load_torchscript` is able to load a file in TorchScript format."""
    test_script_to_torchscript_creates_file(filename)
    model = load_torchscript(filename)
    assert isinstance(model, torch.jit._script.RecursiveScriptModule)
