We want to be able to run ML models directly in Fortran. Initially let's assume that the model has been trained in some other language (say Python) and saved (e.g. as a SavedModel). We want to run inference on this model without having to call the Python executable. This should be possible by using the existing ML C/C++ interfaces.

# PyTorch

PyTorch provides a C++ API (aka LibTorch) that enables inference and online training of pre-trained models in a production environment with no Python dependency. A model's journey from Python to C++ is enabled by TorchScript, an intermediate representation of a PyTorch model. A model trained in Python is first converted to a TorchScript module via tracing or annotation and then serialised to a file. The serialised model can then be loaded in C++ using the C++ API. For more information, see [here](https://pytorch.org/tutorials/advanced/cpp_export.html).

We have created a minimum C wrapper interface of LibTorch (c_torch.h) that allows us to load and infer a TorchScript model. A Fortran interface (mod_torch) has subsequently been built on top of the C interface. For examples using the C++, C and Fortran interfaces to do inference, see the corresponding ts_inference.* file. A Python script is also provided (p2ts.py) that converts a pre-trained ResNet model to TorchScript and serialises it to a file. This model can be used as the input of the ts_infer_* executables for testing.

## Building

These examples have the following pre-requisites:

* [PyTorch](https://pytorch.org/)
* [libtorch](https://pytorch.org/cppdocs/installing.html)

To build, do the following in this directory:

    mkdir build
    cd build
    cmake ../.
    make

### Troubleshooting

if `cmake` has a hard time finding your libtorch install you
can add a line to `CMakeLists.txt` to give a director location, e.g.,
added before `find_package(Torch REQUIRED)`

     set(Torch_DIR /usr/local/lib/libtorch/share/cmake/Torch)

This assumes `libtorch` has been install in `/usr/local/lib`.