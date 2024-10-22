# Example 1 - SimpleNet

This example provides a simple but complete demonstration of how to use the library.

The aim is to demonstrate the most basic features of coupling before worrying about
multi-dimension tensors, row- vs. column-major, multiple inputs etc. that will be
covered in later examples.


## Description

A Python file `simplenet.py` is provided that defines a very simple PyTorch 'net' that takes an input
vector of length 5 and applies a single `Linear` layer to multiply it by 2.

A modified version of the `pt2ts.py` tool saves this simple net to TorchScript.

A series of files `simplenet_infer_<LANG>` then bind from other languages to run the
TorchScript model in inference mode.

## Dependencies

To run this example requires:

- CMake
- Fortran compiler
- FTorch (installed as described in main package)
- Python 3

## Running

To run this example install FTorch as described in the main documentation.
Then from this directory create a virtual environment and install the necessary Python
modules:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You can check that everything is working by running `simplenet.py`:
```
python3 simplenet.py
```
This defines the net and runs it with an input tensor [0.0, 1.0, 2.0, 3.0, 4.0] to produce the result:
```
tensor([[0, 2, 4, 6, 8]])
```

To save the SimpleNet model to TorchScript run the modified version of the
`pt2ts.py` tool:
```
python3 pt2ts.py
```
which will generate `saved_simplenet_model_cpu.pt` - the TorchScript instance of the net.

You can check that everything is working by running the `simplenet_infer_python.py` script:
```
python3 simplenet_infer_python.py
```
This reads the model in from the TorchScript file and runs it with an input tensor
[0.0, 1.0, 2.0, 3.0, 4.0] to produce the result:
```
tensor([[0, 2, 4, 6, 8]])
```

At this point we no longer require Python, so can deactivate the virtual environment:
```
deactivate
```

To call the saved SimpleNet model from Fortran we need to compile the `simplenet_infer`
files.
This can be done using the included `CMakeLists.txt` as follows:
```
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=<path/to/your/installation/of/library/> -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

(Note that the Fortran compiler can be chosen explicitly with the `-DCMAKE_Fortran_COMPILER` flag,
and should match the compiler that was used to locally build FTorch.)

To run the compiled code calling the saved SimpleNet TorchScript from Fortran run the
executable with an argument of the saved model file:
```
./simplenet_infer_fortran ../saved_simplenet_model_cpu.pt
```

This runs the model with the array `[0.0, 1.0, 2.0, 3.0, 4.0]` should produce the output:
```
   0.00000000       2.00000000       4.00000000       6.00000000       8.00000000
```

Alternatively we can use `make`, instead of CMake, with the included Makefile.
However, to do this you will need to modify `Makefile` to link to and include your
installation of FTorch as described in the main documentation. Also check that the compiler is the same as the one you built the Library with.
```
make
./simplenet_infer_fortran saved_simplenet_model_cpu.pt
```

You will also likely need to add the location of the dynamic library files
(`.so` or `.dylib` files) that we will link against at runtime to your `LD_LIBRARY_PATH`:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:</path/to/library/installation>/lib
```
or `DYLD_LIBRARY_PATH` on mac:  
```
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:</path/to/library/installation>/lib
```

## Further options

To explore the functionalities of this model:

- Try saving the model through tracing rather than scripting by modifying `pt2ts.py`
- Consider adapting the model definition in `simplenet.py` to function differently and
  then adapt the rest of the code to successfully couple your new model.
