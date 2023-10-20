---
project: FTorch
summary: A library for coupling (Py)Torch machine learning models to Fortran
author: ICCS Cambridge
license: mit
github: https://github.com/Cambridge-ICCS
project_github: https://github.com/Cambridge-ICCS/FTorch
src_dir: ./src
         ./utils
output_dir: ./doc
exclude_dir: **/build*
extra_filetypes: c   //
                 cpp //
                 h //
                 py  #
sort: alpha
source: true
graph: true
externalize: true
md_extensions: markdown.extensions.toc
               markdown.extensions.tables
               markdown.extensions.fenced_code
---

--------------------

This is the FTorch project.

[TOC]

Brief description
-----------------

It is desirable to be able to run machine learning (ML) models directly in Fortran.
ML models are often trained in some other language (say, Python) using a popular frameworks (say, PyTorch) and saved.
We want to run inference on this model without having to call a Python executable.
To achieve this we use the existing Torch C++ interface, libtorch.

FTorch provides a library enabling a user to directly couple their PyTorch models to Fortran code.
There are also installation instructions for the library and examples of performing coupling.

We support running on both CPU and GPU, and have tested the library on UNIX and Windows based operating systems


License
-------

The FTorch source code, related files and documentation are
distributed under an MIT license.  See the
[LICENSE](d)
file for more details.


Projects using FTorch
---------------------

