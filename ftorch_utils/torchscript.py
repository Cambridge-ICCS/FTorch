"""Module containing utilities for handling TorchScript files."""

import importlib.util
import os
import sys

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


def load_pytorch(
    model_definition_file: str, model_name: str, saved_model_file: str
) -> torch.nn.Module:
    """
    Load a PyTorch model from file.

    Parameters
    ----------
    model_definition_file : str
        name of file containing PyTorch model definition
    model_name : str
        name of the PyTorch model
    saved_model_file : str
        name of file containing saved PyTorch model

    Returns
    -------
    model : torch.NN.Module
        a PyTorch model
    """
    # Import the module containing the model definition
    module_name, _ = os.path.splitext(os.path.basename(model_definition_file))
    module_spec = importlib.util.spec_from_file_location(
        module_name, model_definition_file
    )
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)

    # Construct the PyTorch model and load its weights from file
    cls = getattr(module, model_name)
    model = cls()
    with torch.inference_mode():
        model.load_state_dict(torch.load(saved_model_file, weights_only=True))
    model.eval()
    return model


def load_torchscript(filename: str) -> torch.nn.Module:
    """
    Load a TorchScript model from file.

    Parameters
    ----------
    filename : str
        name of file containing TorchScript model
    """
    return torch.jit.load(filename)
