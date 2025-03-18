# Example 4 - Multiple inputs and multiple outputs

This example revisits the SimpleNet example (again) and demonstrates how to run
a variant of it -- MultiIONet -- for which the 'net' has multiple input tensors
and multiple output tensors.

## Description

A variant of the `simplenet.py` Python file from the first example appears here:
`multiionet.py`. In this case, it defines a simple PyTorch network that takes
*two* input vectors $a$ and $b$ of length 4, concatenates them, and then
multiplies the concatenated vector by the Kronecker product of
$\mathrm{diag}(2,3)$ with the $2\times2$ identity matrix. This has the effect of
scaling the first half of the vector by 2 and the second half by 3. The output
is then split into two components. That is, we have
```math
    \begin{bmatrix} c \\\ d \end{bmatrix}=
    \begin{bmatrix} 2I \\\ & 3I \end{bmatrix}
    \begin{bmatrix} a \\\ b \end{bmatrix}
```
where $a$ and $b$ are inputs and $c$ and $d$ are outputs.

The same `pt2ts.py` tool as in the previous examples is used to save the
network to TorchScript. Again, a series of files `multiionet_infer_<LANG>` then
bind from other languages to run the TorchScript model in inference mode.

## Dependencies

To run this example requires:

- CMake
- FTorch (installed as described in the main package)
- Python 3

## Running

To run this example install FTorch as described in the main documentation. Then
from this directory create a virtual environment and install the necessary
Python modules:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You can check everything is working by running `multiionet.py`:
```
python3 multiionet.py
```
This defines the network and runs it with input tensors [0.0, 1.0, 2.0, 3.0] and
[0.0, -1.0, -2.0, -3.0] to produce the result:
```
(tensor([0., 2., 4., 6.]), tensor([ 0., -3., -6., -9.]))
```

To save the MultiIONet model to TorchScript, run the modified version of the
`pt2ts.py` tool:
```
python3 pt2ts.py
```
which will generate `saved_multiio_model_cpu.pt` - the TorchScript instance of
the network.

You can check everything is working by running the `multiionet_infer_python.py`
script. This reads the model in from the TorchScript file and runs it with the
same inputs as above. The result should again be:
```
(tensor([0., 2., 4., 6.]), tensor([ 0., -3., -6., -9.]))
```

At this point we no longer require Python, so can deactivate the virtual
environment:
```
deactivate
```

To call the saved MultiIONet model from Fortran we need to compile the
`multiionet_infer_fortran.f90` file. This can be done using the included
`CMakeLists.txt` as follows:
```
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=<path/to/your/installation/of/library/> -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

(Note that the Fortran compiler can be chosen explicitly with the `-DCMAKE_Fortran_COMPILER` flag,
and should match the compiler that was used to locally build FTorch.)

To run the compiled code calling the saved MultiIONet TorchScript from Fortran,
run the executable with an argument of the saved model file:
```
./multiionet_infer_fortran ../saved_multiio_model_cpu.pt
```
This runs the model with the same inputs as above and should produce the output:
```
   0.00000000       2.00000000       4.00000000       6.00000000    
   0.00000000      -3.00000000      -6.00000000      -9.00000000
```
