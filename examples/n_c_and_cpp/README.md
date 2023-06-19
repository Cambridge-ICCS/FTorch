# Example n - ResNet-18 for c and c++

**Note: This Example is not currently functional and is still being developed.**

This example provides a demonstration of how to use the library to interface to c and c++.

## Description

A python file is provided that downloads the pretrained
[ResNet-18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
model from [TorchVision](https://pytorch.org/vision/stable/index.html).

A modified version of the `pt2ts.py` tool saves this ResNet-18 to TorchScript.

A series of files `resnet_infer_<LANG>` then bind from other languages to run the
TorchScript ResNet-18 model in inference mode.

## Dependencies

To run this example requires:

- cmake
- c compiler
- c++ compiler
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

You can check that everything is working by running `resnet18.py`:
```
python3 resnet18.py
```
it should produce the result `tensor([[623, 499, 596, 111, 813]])`.

To save the pretrained ResNet-18 model to TorchScript run the modified version of the
`pt2ts.py` tool :
```
python3 pt2ts.py
```

At this point we no longer require python, so can deactivate the virtual environment:
```
deactivate
```

To call the saved ResNet-18 model from fortran we need to compile the `resnet_infer`
files.
This can be done using the included `CMakeLists.txt` as follows:
```
mkdir build
cd build
cmake .. -DFTorchDIR=<path/to/your/installation/of/library> -DCMAKE_BUILD_TYPE=Release
make
```

To run the compiled code calling the saved ResNet-18 TorchScript from Fortran:
```
./resnet_infer_c
./resnet_infer_cpp
```

### Summary of files

The starting point for understanding the examples is
one of the ts_inference.* files, depending on which
language you wish

* ts_inference.c   - Example of doing inference from C
* ts_inference.cpp - Example of doing inference from C++
* ts_inference.f90 - Example of doing inference from Fortran
* ts_inference.py  - Example of doing inference from Python

* pt2ts.py - Python code to train model

* ctorch.cpp - Wrapper onto PyTorch for C++
* ctorch.h   - Header file for C++ wrapper
* ftorch.f90 - Wrapper onto PyTorch for Fortran

* CMakeLists.txt - Provides the cmake build system configuration



    mkdir build
    cd build
    cmake ../.
    make

This will build the separate examples. Next, from this directory, you need
to run the python code to generate the saved model:

    python3 ../pt2ts.py

This will give you some options about how you want to train the model, and
will output a saved model as a .pt file. You should then be able to run the test program (written in Fortran and now compiled) in this directory:

    ./ts_infer_fortran

which will return a result of querying the model, e.g:

    $ ./ts_infer_fortran
    -1.0526 -0.4629 -0.4567 -1.0881 -0.7655
    [ CPUFloatType{1,5} ]

Note that the Fortran example have the model filename hard-coded (e.g.
"annotated_cpu.pt" as the default). Change this and re-compile to address
the other model outputs.
