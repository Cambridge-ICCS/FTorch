# Example n - Optimisers

**This example is currently under development.** Eventually, it will demonstrate
the use of optimisers in FTorch by leveraging PyTorch's optim module.

By exposing optimisers in Fortran, FTorch will be able to compute optimisation
steps to update models as part of a training process.

## Description

A Python demo is copied from the PyTorch documentation as `optimisers.py`, which
shows how to use an optimiser in PyTorch.

The demo will be replicated in Fortran as `optimisers.f90`, to show how to do the
same thing using FTorch.

## Dependencies

To run this example requires:

- CMake
- Fortran compiler
- FTorch (installed as described in main package)
- Python 3

## Running

To run this example install FTorch as described in the main documentation.
Then from this directory create a virtual environment and install the necessary
Python modules:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the Python version of the demo with
```
python3 optimisers.py
```
This trains a tensor to scale, elementwise, a vector of ones to the vector `[1, 2, 3, 4]`.
It uses the torch SGD optimiser to adjust the values of the scaling tensor at each step,
outputting values of interest to screen in the form:
```console
========================
Epoch: 0
        Output:
                tensor([1., 1., 1., 1.], grad_fn=<MulBackward0>)
        loss:
                3.5
        tensor gradient:
                tensor([ 0.0000, -0.5000, -1.0000, -1.5000])
        tensor:
                tensor([1.0000, 1.5000, 2.0000, 2.5000], requires_grad=True)
...
```
