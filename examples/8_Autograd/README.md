# Example 8 - Autograd

The worked examples in this subdirectory demonstrate automatic differentiation
in FTorch by leveraging PyTorch's Autograd module.

By exposing Autograd in Fortran, FTorch is able to compute derivatives of
expressions involving `torch_tensor`s.

## Description

First, we include a modified version of the Python demo found in the PyTorch
documentation as `autograd.py`, which shows how to compute the gradient of a
mathematical expression involving Torch Tensors. The demo is replicated in
Fortran as `autograd.f90`, to show how to do the same thing using FTorch.

Second, we provide `simplenet.py`, which defines a simple neural network using
PyTorch's `nn.Module` class, saves it in `TorchScript` format, and shows how to
differentiate through the network propagation in Python. The Fortran version,
`simplenet.f90`, shows how to load the saved model and differentiate through a
call to `torch_model_forward` in FTorch.

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

### Maths demo

Run the Python version of the first demo with
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
This should print
```
 dQ/da = 9*a^2 =    36.0000000       81.0000000
 dQ/db = - 2*b =   -12.0000000      -8.00000000
```
along with some testing output.

### Neural network demo

The second example proceeds in much the same way, replacing `autograd` with
`simplenet`. Running
```
python3 simplenet.py
```
should give the output
```
y = tensor([[ 2.,  4.,  6.,  8., 10.]], grad_fn=<MmBackward0>)
dy/dx = tensor([[2., 2., 2., 2., 2.]])
```

To run the Fortran version, simply execute
```
./simplenet
```
This should
