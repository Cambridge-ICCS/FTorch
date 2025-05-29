# Example 6 - Autograd

This example demonstrates automatic differentation in FTorch by leveraging
PyTorch's Autograd module.

By exposing Autograd in Fortran, FTorch is able to compute derivatives of
expressions involving `torch_tensor`s.

## Description

A modified version of the Python demo found in the PyTorch documentation as
`autograd.py`, which shows how to compute the gradient of a mathematical
expression involving Torch Tensors.

The demo is replicated in Fortran as `autograd.f90`, to show how to do the same
thing using FTorch.

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
python3 autograd.py
```
This performs some arithmetic on two input tensors [2.0, 3.0] and [6.0, 4.0] to
produce the result:
```
tensor([-12., 65.], grad_fn=<SubBackward0>)
```
where `<SubBackward0>` refers to the method used for computing the gradient.


To run the Fortran version of the demo we need to compile with (for example)
```
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=<path/to/your/installation/of/library/> -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

(Note that the Fortran compiler can be chosen explicitly with the `-DCMAKE_Fortran_COMPILER` flag,
and should match the compiler that was used to locally build FTorch.)

To run the compiled code, simply use
```
./autograd
```
