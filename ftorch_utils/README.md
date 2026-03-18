# FTorch: Utils

This directory contains useful Python utilities for users of the library.

## `pt2ts.py`

This is a Python script that can take a PyTorch model and convert it to
TorchScript.
It provides the user with the option to [jit.script](https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch.jit.script) or [jit.trace](https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch.jit.trace).

Dependencies:
- PyTorch

### Usage

1. Create and activate a virtual environment.
2. Install `ftorch_utils` and its dependencies, if they aren't already
   installed. This can be achieved by running `pip install .` in the root
   directory of FTorch. It will install the `pt2ts` script into your Python
   environment.
3. Run with `python3 -m pt2ts <arguments>` or simply `pt2ts <arguments>`.
   Run `pt2ts --help` (or below) to see the usage instructions for the script.

The Torchscript model will be saved locally in the specified location.

#### Command line arguments

The `pt2ts.py` script is set up with the `argparse` Python module such that it
accepts several command line arguments:
* `model_definition_file </path/to/model/def>`, which specifies the path of
  the local file in which a PyTorch model is defined.
* `model_name <model_name>`, which is required to specify the name of the model
  class (e.g. `SimpleNet`).
* `input_model_file </path/to/saved/model>`, which is required to specify the
  path to the directory in which the PyTorch model is saved.
* `--output_model_file </path/to/save/model>`, which allows you to specify the
  path to the directory in which the TorchScript model should be saved.
* `--trace`, which allows you to switch from scripting to tracing.
* `--test`, which allows you to run basic tests to check things are working as
  expected.
* `--input_tensor_file </path/to/saved/tensor>`, which is required (if `--trace`
  or `--test` was passed) to specify the path to the directory in which a
  PyTorch tensor of appropriate input dimensions is saved.
