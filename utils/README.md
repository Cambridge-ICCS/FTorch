# FTorch: Utils

This directory contains useful utilities for users of the library.

## `pt2ts.py`

This is a python script that can take a PyTorch model and convert it to Torchscript.
It provides the user with the option to [jit.script](https://pytorch.org/docs/stable/generated/torch.jit.script.html#torch.jit.script) or [jit.trace](https://pytorch.org/docs/stable/generated/torch.jit.trace.html#torch.jit.trace) for both CPU and GPU.

Dependencies:
- PyTorch

### Usage
1. Create and activate a virtual environment with PyTorch and any dependencies for your model.
2. Place the `pt2ts.py` script in the same folder as your model files.
3. Import your model into `pt2ts.py` and amend options as necessary (search for `FPTLIB-TODO`).
4. Run with `python3 pt2ts.py`.

The model will be saved in the location from which `pt2ts.py` is running.
