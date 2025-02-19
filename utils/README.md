# FTorch: Utils

This directory contains useful utilities for users of the library.

## `pt2ts.py`

This is a python script that can take a PyTorch model and convert it to
TorchScript.
It provides the user with the option to [jit.script](https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch.jit.script) or [jit.trace](https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch.jit.trace) for both CPU and GPU.

Dependencies:
- PyTorch

### Usage

1. Create and activate a virtual environment with PyTorch and any dependencies for your model.
2. Place the `pt2ts.py` script in the same folder as your model files.
3. Import your model into `pt2ts.py` and amend options as necessary (search for `FPTLIB-TODO`).
4. Run with `python3 pt2ts.py`.

The Torchscript model will be saved locally in the same location from which the `pt2ts.py`
script is being run.

#### Command line arguments

The `pt2ts.py` script is set up with the `argparse` Python module such that it
accepts two command line arguments:
* `--filepath </path/to/save/model>`, which allows you to specify the path
  to the directory in which the TorchScript model should be saved.
* `--device_type <cpu/cuda/xpu/mps>`, which allows you to specify the CPU or GPU
  device type with which to save the model. (To read and make use of a model on
  a particular architecture, it's important that the model was targeted for that
  architecture when it was saved.)
