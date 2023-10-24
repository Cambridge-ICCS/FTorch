---
project: FTorch
summary: A library for coupling (Py)Torch machine learning models to Fortran
author: ICCS Cambridge
license: mit
github: https://github.com/Cambridge-ICCS
project_github: https://github.com/Cambridge-ICCS/FTorch
page_dir: pages
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

Presentations
-------------

The following presentations contain information about FTorch:

* Reducing the Overhead of Coupled Machine Learning Models between Python and Fortran  
  RSECon23  
  [Slides](https://jackatkinson.net/slides/RSECon23/RSECon23.html)


License
-------

The FTorch source code, related files and documentation are
distributed under an [MIT License which can be viewed here](page/LICENSE.html).


Projects using FTorch
---------------------

The following projects make use of FTorch.  
If you use our library in your work please let us know.

* [MiMA Machine Learning - DataWave](https://github.com/DataWaveProject/MiMA-machine-learning)  
  Using FTorch to couple a neural net parameterisation of gravity waves to an atmospheric model.
