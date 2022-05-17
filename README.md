# fortran-ml-bridge
Code and examples on directly calling Tensorflow / Pytorch ML models from Fortran.

## Problem Statement
We want to be able to run ML models directly in Fortran. Initially let's assume that the model has been trained in some other language (say Python) and saved (e.g. as a SavedModel). We want to run inference on this model without having to call the Python executable. This should be possible by using the existing ML C/C++ interfaces.

### PyTorch

See the [fortran-pytorch-lib](https://github.com/Cambridge-ICCS/fortran-ml-bridge/tree/main/fortran-pytorch-lib) directory.

### Tensorflow

See the [fortran-tf-lib](https://github.com/Cambridge-ICCS/fortran-ml-bridge/tree/main/fortran-tf-lib) directory.