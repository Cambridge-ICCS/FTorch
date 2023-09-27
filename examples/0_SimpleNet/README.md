# Example 0 - SimpleNet

This example provides a simple but complete demonstration of how to use the library.

The aim is to demonstrate the most basic features of coupling before worrying about
multi-dimension tensors, row- vs. column-major, multiple inputs etc. that will be
covered in later examples.


## Description

A python file `simplenet.py` is provided that defines a very simple pytorch 'net' that takes an input
vector of length 5 and applies a single `Linear` layer to multiply it by 2.

A modified version of the `pt2ts.py` tool saves this simple net to TorchScript.

A series of files `simplenet_infer_<LANG>` then bind from other languages to run the
TorchScript model in inference mode.

## Dependencies

To run this example requires:

- cmake
- fortran compiler
- FTorch (installed as described in main package)
- python3

## Running

To run this example install fortran-pytorch-lib as described in the main documentation.
Then from this directory create a virtual environment an install the neccessary python
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
it should produce the result `tensor([[0, 2, 4, 6, 8]])`.

To save the SimpleNet model to TorchScript run the modified version of the
`pt2ts.py` tool :
```
python3 pt2ts.py
```

At this point we no longer require python, so can deactivate the virtual environment:
```
deactivate
```

To call the saved SimpleNet model from fortran we need to compile the `simplenet_infer`
files.
This can be done using the included `CMakeLists.txt` as follows:
```
mkdir build
cd build
cmake .. -DFTorch_DIR=<path/to/your/installation/of/library/>lib/cmake/ -DCMAKE_BUILD_TYPE=Release
make
```
Make sure that the  `FTorch_DIR` flag points to the `lib/cmake/` folder within the installation of the FTorch library.  

To run the compiled code calling the saved SimpleNet TorchScript from Fortran run the
executable with an argument of the saved model file:
```
./simplenet_infer_fortran ../saved_simplenet_model_cpu.pt
```

Alternatively we can use `make`, instead of cmake, with the included Makefile.
However, to do this you will need to modify `Makefile` to link to and include your
installation of FTorch as described in the main documentation. Also check that the compiler is the same as the one you built the Library with.
```
make
./simplenet_infer_fortran saved_simplenet_model_cpu.pt
```

You will also likely need to add the location of the `.so` or `.dylib` files to your `LD_LIBRARY_PATH`:
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
