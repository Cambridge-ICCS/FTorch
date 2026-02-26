# FTorch: Utils

This directory contains useful Python utilities for users of the library.

## `pt2ts.py`

This is a python script that can take a PyTorch model and convert it to
TorchScript.
It provides the user with the option to [jit.script](https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch.jit.script) or [jit.trace](https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch.jit.trace).

Dependencies:
- PyTorch

### Usage

1. Create and activate a virtual environment with PyTorch and any dependencies
   for your model.
2. Place the `pt2ts.py` script in the same folder as your model files.
3. Run with `python3 pt2ts.py` or `./pt2ts.py`.

The Torchscript model will be saved locally in the same location from which the `pt2ts.py`
script is being run.

#### Command line arguments

The `pt2ts.py` script is set up with the `argparse` Python module such that it
accepts several command line arguments:
* `input_model_file </path/to/saved/model>`, which is required to specify the
  path to the directory in which the PyTorch model is saved.
* `--output_model_file </path/to/save/model>`, which allows you to specify the
  path to the directory in which the TorchScript model should be saved.
* `--trace`, which allows you to switch from scripting to tracing.
* `--input_tensor_file </path/to/saved/tensor>`, which is required (if `--trace`
  was passed) to specify the path to the directory in which a PyTorch tensor of
  appropriate input dimensions is saved.
