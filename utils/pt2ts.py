"""Load a PyTorch model and convert it to TorchScript."""
# Throughout this script there are various `FTORCH-TODO` comments indicating where
# the user needs to modify as appropriate for their model

import os
from typing import Optional

# FTORCH-TODO
# Add a module import with your model here:
# This example assumes the model architecture is in an adjacent module `my_ml_model.py`
import my_ml_model
import torch


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
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device_type",
        help="Device type to run the inference on",
        type=str,
        choices=["cpu", "cuda", "hip", "xpu", "mps"],
        default="cpu",
    )
    parser.add_argument(
        "--filepath",
        help="Path to the file containing the PyTorch model",
        type=str,
        default=os.path.dirname(__file__),
    )
    parsed_args = parser.parse_args()
    device_type = parsed_args.device_type
    filepath = parsed_args.filepath

    # =====================================================
    # Load model and prepare for saving
    # =====================================================

    # FTORCH-TODO
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

    # FTORCH-TODO
    # Generate a dummy input Tensor `dummy_input` to the model of appropriate size.
    # This example assumes two inputs of size (512x40) and (512x1)
    trained_model_dummy_input_1 = torch.ones((512, 40), dtype=torch.float64)
    trained_model_dummy_input_2 = torch.ones((512, 1), dtype=torch.float64)

    # Transfer the model and inputs to GPU device, if appropriate
    if device_type != "cpu":
        if device_type == "hip":
            device = torch.device("cuda")  # NOTE: HIP is treated as CUDA in FTorch
        else:
            device = torch.device(device_type)
        trained_model = trained_model.to(device)
        trained_model.eval()
        trained_model_dummy_input_1 = trained_model_dummy_input_1.to(device)
        trained_model_dummy_input_2 = trained_model_dummy_input_2.to(device)

    # FTORCH-TODO
    # Run model for dummy inputs
    # If something isn't working This will generate an error
    trained_model_dummy_outputs = trained_model(
        trained_model_dummy_input_1,
        trained_model_dummy_input_2,
    )

    # =====================================================
    # Save model
    # =====================================================

    # FTORCH-TODO
    # Set the name of the file you want to save the torchscript model to:
    saved_ts_filename = f"saved_model_{device_type}.pt"
    # A filepath may also be provided. To do this, pass the filepath as an argument to
    # this script when it is run from the command line, i.e. `./pt2ts.py path/to/model`.

    # FTORCH-TODO
    # Save the PyTorch model using either scripting (recommended if possible) or tracing
    # -----------
    # Scripting
    # -----------
    script_to_torchscript(trained_model, filename=saved_ts_filename)

    # -----------
    # Tracing
    # -----------
    # trace_to_torchscript(
    #     trained_model, trained_model_dummy_input, filename=saved_ts_filename
    # )

    # =====================================================
    # Check model saved OK
    # =====================================================

    # Load torchscript and run model as a test, scaling inputs as above
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
            model_error = (
                "Saved Torchscript model is not performing as expected.\n"
                "Consider using scripting if you used tracing, or investigate further."
            )
            raise RuntimeError(model_error)

    # Check that the model file is created
    if not os.path.exists(os.path.join(filepath, saved_ts_filename)):
        torchscript_file_error = (
            f"Saved TorchScript file {os.path.join(filepath, saved_ts_filename)} "
            "cannot be found."
        )
        raise FileNotFoundError(torchscript_file_error)
