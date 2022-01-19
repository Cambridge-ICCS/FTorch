# fortran-ml-bridge
Code and examples on directly calling Tensorflow / Pytorch / Keras ML models from Fortran.

## The problem
We want to be able to run models directly from Fortran. Initially let's assume that
the model has been trained in some other language (say Python) and saved (e.g. as a SavedModel).
We want to run inference on this model without having to call the python executable. This should be
possible by using the existing ML C/C++ interfaces.
