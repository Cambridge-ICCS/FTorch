"""Load a PyTorch model and convert it to TorchScript."""

import os
import sys
from typing import Optional
import torch

# FPTLIB-TODO
# Add a module import with your model here:
# This example assumes the model architecture is in an adjacent module `my_ml_model.py`
import my_ml_model


def script_to_torchscript(
    model: torch.nn.Module, filename: Optional[str] = "scripted_model.pt"
) -> None:
    """
    Save PyTorch model to TorchScript using scripting.

    Parameters
    ----------
    model : torch.NN.Module
        a PyTorch model
    filename : str
        name of file to save to
    """
    # FIXME: torch.jit.optimize_for_inference() when PyTorch issue #81085 is resolved
    scripted_model = torch.jit.script(model)
    # print(scripted_model.code)
    scripted_model.save(filename)


def trace_to_torchscript(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    filename: Optional[str] = "traced_model.pt",
) -> None:
    """
    Save PyTorch model to TorchScript using tracing.

    Parameters
    ----------
    model : torch.NN.Module
        a PyTorch model
    dummy_input : torch.Tensor
        appropriate size Tensor to act as input to model
    filename : str
        name of file to save to
    """
    # FIXME: torch.jit.optimize_for_inference() when PyTorch issue #81085 is resolved
    traced_model = torch.jit.trace(model, dummy_input)
    # traced_model.save(filename)
    frozen_model = torch.jit.freeze(traced_model)
    ## print(frozen_model.graph)
    ## print(frozen_model.code)
    frozen_model.save(filename)


def load_torchscript(filename: Optional[str] = "saved_model.pt") -> torch.nn.Module:
    """
    Load a TorchScript from file.

    Parameters
    ----------
    filename : str
        name of file containing TorchScript model
    """
    model = torch.jit.load(filename)

    return model


if __name__ == "__main__":
    # =====================================================
    # Load model and prepare for saving
    # =====================================================

    # FPTLIB-TODO
    # Load a pre-trained PyTorch model
    # Insert code here to load your model as `trained_model`.
    # This example assumes my_ml_model has a method `initialize` to load
    # architecture, weights, and place in inference mode
    trained_model = my_ml_model.initialize()

    # Switch off specific layers/parts of the model that behave
    # differently during training and inference.
    # This may have been done by the user already, so just make sure here.
    trained_model.eval()

    # =====================================================
    # Prepare dummy input and check model runs
    # =====================================================

    # FPTLIB-TODO
    # Generate a dummy input Tensor `dummy_input` to the model of appropriate size.
    # This example assumes two inputs of size (512x40) and (512x1)
    trained_model_dummy_input_1 = torch.ones((512, 40), dtype=torch.float64)
    trained_model_dummy_input_2 = torch.ones((512, 1), dtype=torch.float64)

    # FPTLIB-TODO
    # Uncomment the following lines to save for inference on GPU (rather than CPU):
    # device = torch.device('cuda')
    # trained_model = trained_model.to(device)
    # trained_model.eval()
    # trained_model_dummy_input_1 = trained_model_dummy_input_1.to(device)
    # trained_model_dummy_input_2 = trained_model_dummy_input_2.to(device)

    # FPTLIB-TODO
    # Run model for dummy inputs
    # If something isn't working This will generate an error
    trained_model_dummy_outputs = trained_model(
        trained_model_dummy_input_1,
        trained_model_dummy_input_2,
    )

    # =====================================================
    # Save model
    # =====================================================

    # FPTLIB-TODO
    # Set the name of the file you want to save the torchscript model to:
    saved_ts_filename = "saved_model.pt"
    # A filepath may also be provided. To do this, pass the filepath as an argument to
    # this script when it is run from the command line, i.e., `./pt2ts.py path/to/model`.

    # FPTLIB-TODO
    # Save the PyTorch model using either scripting (recommended where possible) or tracing
    # -----------
    # Scripting
    # -----------
    script_to_torchscript(trained_model, filename=saved_ts_filename)

    # -----------
    # Tracing
    # -----------
    # trace_to_torchscript(trained_model, trained_model_dummy_input, filename=saved_ts_filename)

    # =====================================================
    # Check model saved OK
    # =====================================================

    # Load torchscript and run model as a test
    # FPTLIB-TODO
    # Scale inputs as above and, if required, move inputs and mode to GPU
    trained_model_dummy_input_1 = 2.0 * trained_model_dummy_input_1
    trained_model_dummy_input_2 = 2.0 * trained_model_dummy_input_2
    trained_model_testing_outputs = trained_model(
        trained_model_dummy_input_1,
        trained_model_dummy_input_2,
    )
    ts_model = load_torchscript(filename=saved_ts_filename)
    ts_model_outputs = ts_model(
        trained_model_dummy_input_1,
        trained_model_dummy_input_2,
    )

    if not isinstance(ts_model_outputs, tuple):
        ts_model_outputs = (ts_model_outputs,)
    if not isinstance(trained_model_testing_outputs, tuple):
        trained_model_testing_outputs = (trained_model_testing_outputs,)
    for ts_output, output in zip(ts_model_outputs, trained_model_testing_outputs):
        if torch.all(ts_output.eq(output)):
            print("Saved TorchScript model working as expected in a basic test.")
            print("Users should perform further validation as appropriate.")
        else:
            raise RuntimeError(
                "Saved Torchscript model is not performing as expected.\n"
                "Consider using scripting if you used tracing, or investigate further."
            )

    # Check that the model file is created
    filepath = os.path.dirname(__file__) if len(sys.argv) == 1 else sys.argv[1]
    assert os.path.exists(os.path.join(filepath, saved_ts_filename))
