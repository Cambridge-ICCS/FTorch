# Example 3 - MultiGPU

This example revisits the SimpleNet example and demonstrates how to run it using
multiple GPU devices.


## Description

The same python file `simplenet.py` is used from the earlier example. Recall that it
defines a very simple PyTorch network that takes an input of length 5 and applies a
single `Linear` layer to multiply it by 2.

The same `pt2ts.py` tool is used to save the simple network to TorchScript.

A series of files `simplenet_infer_<LANG>` then bind from other languages to run the
TorchScript model in inference mode.

## Dependencies

To run this example requires:

- cmake
- mpif90
- FTorch (installed as described in main package with mpicc, mpicxx, and mpif90)
- python3

## Running

To run this example install FTorch as described in the main documentation. Then from
this directory create a virtual environment and install the necessary python modules:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy over the `simplenet.py` script from the earlier example:
```
cp ../1_SimpleNet/simplenet.py .
```

You can check that everything is working by running `simplenet.py`:
```
python3 simplenet.py
```
As before, this defines the network and runs it with an input tensor
[0.0, 1.0, 2.0, 3.0, 4.0] to produce the result:
```
tensor([[0, 2, 4, 6, 8]])
```

To save the SimpleNet model to TorchScript run the modified version of the `pt2ts.py`
tool:
```
python3 pt2ts.py
```
which will generate `saved_simplenet_model_cuda.pt` - the TorchScript instance of the
network. The only difference with the earlier example is that the model is built to
be run using CUDA rather than on CPU.

You can check that everything is working by running the `simplenet_infer_python.py`
script. It's set up with MPI such that a different GPU device is associated with each
MPI rank. If you have four GPUs available then use the following, otherwise adjust the
number of MPI ranks.
```
mpiexec -np 4 python3 simplenet_infer_python.py
```
This reads the model in from the TorchScript file and runs it with an different input
tensor on each GPU device: [0.0, 1.0, 2.0, 3.0, 4.0], plus the device index in each
entry. The result should be (some permutation of):
```
0: tensor([[0, 2, 4, 6, 8]])
1: tensor([[2, 4, 6, 8, 10]])
2: tensor([[4, 6, 8, 10, 12]])
3: tensor([[6, 8, 10, 12, 14]])
```

At this point we no longer require python, so can deactivate the virtual environment:
```
deactivate
```

To call the saved SimpleNet model from Fortran we need to compiler the `simplnet_infer`
files. This can be done using the included `CMakeLists.txt` as follows, noting that we
need to use an MPI-enabled Fortran compiler:
```
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=<path/to/your/installation/of/library/> -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

To run the compiled code calling the saved SimpleNet TorchScript from Fortran, run the
executable with an argument of the saved model file. Again, adjust the number of MPI
processes, if required:
```
mpiexec -np 4 ./simplenet_infer_fortran ../saved_simplenet_model_cuda.pt
```

# TODO: Running

# TODO: Make rather than CMake

# TODO: Further options
