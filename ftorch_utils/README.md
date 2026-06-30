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
   environment allowing you to call it directly as `pt2ts` when the
   virtual environment is active.
3. Run with `python3 -m pt2ts <arguments>` or simply `pt2ts <arguments>`.
   Run `pt2ts --help` (or below) to see the usage instructions for the script.

The Torchscript model will be saved locally in the specified location.

#### Command line arguments

The `pt2ts.py` script is set up with the `argparse` Python module such that it
accepts several command line arguments:
* `model_name <model_name>`, which is required to specify the name of the model
  class (e.g. `SimpleNet`). This can also be used to load pre-trained models
  from TorchVision by providing the name used in `torchvision.models`.
* `--model_definition_file </path/to/model/def>`, which specifies the file name
  for a local PyTorch model definition (including its path). (Not required for
  pre-trained models.)
* `--input_model_file </path/to/saved/model>`, which specifies the file name of
  a local PyTorch model file (including its path). (Not required for pre-trained
  models.)
* `--output_model_file </path/to/save/model>`, which specifies the file name of
  the TorchScript model to be written out (including its path).
* `--trace`, which allows you to switch from scripting to tracing.
* `--test`, which allows you to run basic tests to check things are working as
  expected.
* `--precision`, which specifies the working precision for the model and all
  PyTorch operations, e.g., 'float32'.
* `--model_weights`, which allows customisation of model weights for pre-trained
  models or locally defined models whose `__init__` method accepts a
  `model_weights` keyword argument.
* `--input_tensor_file </path/to/saved/tensor>`, which is required (if `--trace`
  or `--test` was passed) to specify the file name of a local PyTorch tensor of
  appropriate input dimensions (including its path).
