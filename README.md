# fortran-pytorch-lib
Code and examples on directly calling Pytorch ML models from Fortran.

## Problem Statement
We want to be able to run ML models directly in Fortran. Initially let's assume
that the model has been trained in some other language (say Python) and saved.
We want to run inference on this model without having
to call the Python executable. This should be possible by using the existing ML
C/C++ interfaces.
