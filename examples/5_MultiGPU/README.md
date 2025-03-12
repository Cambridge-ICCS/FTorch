# Example 3 - MultiGPU

This example revisits the SimpleNet example and demonstrates how to run it using
multiple GPU devices.


## Description

The same Python file `simplenet.py` is used from the earlier example. Recall
that it defines a very simple PyTorch network that takes an input of length 5
and applies a single `Linear` layer to multiply it by 2.

The same `pt2ts.py` tool is used to save the simple network to TorchScript.

A series of files `multigpu_infer_<LANG>` then bind from other languages to run
the TorchScript model in inference mode.

## Dependencies

To run this example requires:

- CMake
- Two (or more) CUDA or XPU GPU devices (or a single MPS device).
- FTorch (installed with a GPU_DEVICE enabled as described in main package)
- Python 3

## Running

To run this example install FTorch as described in the main documentation. Then from
this directory create a virtual environment and install the necessary Python modules:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You can check that everything is working by running `simplenet.py`:
```
python3 simplenet.py --device_type <my_device_type>
```
where `<my_device_type>` is `cuda`/`xpu`/`mps` as appropriate for your device.

As before, this defines the network and runs it with an input tensor
[0.0, 1.0, 2.0, 3.0, 4.0]. The difference is that the code will make use of the
default GPU device (index 0) to produce the result:
```
SimpleNet forward pass on CUDA device 0
tensor([[0, 2, 4, 6, 8]])
```
for CUDA, and similarly for other device types.

To save the `SimpleNet` model to TorchScript run the modified version of the
`pt2ts.py` tool:
```
python3 pt2ts.py --device_type <my_device_type>
```
which will generate `saved_multigpu_model_<my_device_type>.pt` - the TorchScript
instance of the network. The only difference with the earlier example is that
the model is built to be run on GPU devices rather than on CPU.

You can check that everything is working by running the
`multigpu_infer_python.py` script. It's set up such that it loops over two GPU
devices. Run with:
```
python3 multigpu_infer_python.py --device_type <my_device_type>
```
This reads the model in from the TorchScript file and runs it with an different input
tensor on each GPU device: [0.0, 1.0, 2.0, 3.0, 4.0], plus the device index in each
entry. The result should be:
```
Output on device 0: tensor([[0., 2., 4., 6., 8.]])
Output on device 1: tensor([[ 2., 4.,  6.,  8., 10.]])
```
Note that Mps will only use device 0.

At this point we no longer require Python, so can deactivate the virtual environment:
```
deactivate
```

To call the saved `SimpleNet` model from Fortran we need to compile the
`multigpu_infer_fortran.f90` file. This can be done using the included
`CMakeLists.txt` as follows, noting that we need to use an MPI-enabled Fortran
compiler:
```
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=<path/to/your/installation/of/library/> -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

(Note that the Fortran compiler can be chosen explicitly with the `-DCMAKE_Fortran_COMPILER` flag,
and should match the compiler that was used to locally build FTorch.)

To run the compiled code calling the saved `SimpleNet` TorchScript from
Fortran, run the executable with arguments of device type and the saved model file:
```
./multigpu_infer_fortran <cuda/xpu/mps> ../saved_multigpu_model_<cuda/xpu>.pt
```

This runs the model with the same inputs as described above and should produce (some
permutation of) the output:
```
input on device 0: [  0.0,  1.0,  2.0,  3.0,  4.0]
input on device 1: [  1.0,  2.0,  3.0,  4.0,  5.0]
output on device 0: [  0.0,  2.0,  4.0,  6.0,  8.0]
output on device 1: [  2.0,  4.0,  6.0,  8.0, 10.0]
```
Again, note that MPS will only use device 0.

Alternatively, we can use `make`, instead of CMake, copying the Makefile over from the
first example:
```
cp ../1_SimpleNet/Makefile .
```
See the instructions in that example directory for further details.
