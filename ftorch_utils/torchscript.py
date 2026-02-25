"""Module containing utilities for handling TorchScript files."""

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
