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
preprocess: true
macro: UNIX
       GPU_DEVICE_NONE=0
       GPU_DEVICE_CUDA=1
       GPU_DEVICE_XPU=11
       GPU_DEVICE_MPS=12
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
To achieve this we use the existing Torch C++ interface, LibTorch.

FTorch provides a library enabling a user to directly couple their PyTorch models to Fortran code.
There are also installation instructions for the library and examples of performing coupling.

We support running on both CPU and GPU, and have tested the library on UNIX and Windows based operating systems


News
----

Some recent updates about FTorch:

* FTorch developers Jack Atkinson and Joe Wallwork
  [were recently awarded funding from C2D3-Accelerate](https://science.ai.cam.ac.uk/news/2024-12-09-exploring-novel-applications-of-ai-for-research-and-innovation-%E2%80%93-announcing-our-2024-funded-projects.html)
  for a project entitled *Online training of large-scale Fortran-based hybrid
  computational science models, with applications in climate science*. This
  grant will support development for online training functionality for FTorch,
  as well as research visits, FTorch training tutorials, and organisation of a
  [workshop on hybrid modelling](https://cambridge-iccs.github.io/ml-coupling-workshop).
* FTorch developer Joe Wallwork will be giving a talk at the 2025
  [EuroAD workshop](https://scicomp.rptu.de/27th-euroad-workshop/) on 3rd-4th
  April 2025 entitled *Facilitating online training in Fortran-based climate
  models*.
* FTorch developers Joe Wallwork and Tom Meltzer will be giving an FTorch
  tutorial at
  [Durham HPC Days](https://www.durham.ac.uk/research/institutes-and-centres/data-science/events-/durham---hpc-days/)
  on 2nd June 2025.


Publications and Presentations
------------------------------

FTorch is published in JOSS. To cite it in your work please refer to:

Atkinson et al., (2025). FTorch: a library for coupling PyTorch models to Fortran.
_Journal of Open Source Software_, 10(107), 7602, [https://doi.org/10.21105/joss.07602](https://doi.org/10.21105/joss.07602)

The following presentations contain information about FTorch:

* Coupling Machine Learning to Numerical (Climate) Models<br>
  Platform for Advanced Scientific Computing, Zurich - June 2024<br>
  [Slides](https://jackatkinson.net/slides/PASC24)
* Blending Machine Learning and Numerical Simulation, with Applications to Climate Modelling<br>
  Durham HPC days, Durham - May 2024<br>
  [Slides](https://jackatkinson.net/slides/HPC_Durham_2024)
* Reducing the overheads for coupling PyTorch machine learning models to Fortran<br>
  ML & DL Seminars, LSCE, IPSL, Paris - November 2023<br>
  [Slides](https://jackatkinson.net/slides/IPSL_FTorch) - [Recording](https://www.youtube.com/watch?v=-NJGuV6Rz6U)
* Reducing the Overhead of Coupled Machine Learning Models between Python and Fortran<br>
  RSECon23, Swansea - September 2023<br>
  [Slides](https://jackatkinson.net/slides/RSECon23) - [Recording](https://www.youtube.com/watch?v=Ei6H_BoQ7g4&list=PL27mQJy8eDHmibt_aL3M68x-4gnXpxvZP&index=33)


License
-------

The FTorch source code, related files and documentation are
distributed under an [MIT License which can be viewed here](page/LICENSE.html).


Projects using FTorch
---------------------

The following projects make use of FTorch.  
If you use our library in your work please let us know.

* [DataWave CAM-GW](https://github.com/DataWaveProject/CAM/) -
  Using FTorch to couple neural net parameterisations of gravity waves to the CAM
  atmospheric model.
* [MiMA Machine Learning](https://github.com/DataWaveProject/MiMA-machine-learning) -
  Implementing a neural net parameterisation of gravity waves in the MiMA atmospheric model.
  Demonstrates that nets trained near-identically offline can display greatly varied behaviours when coupled online.
  See Mansfield and Sheshadri (2024) - [DOI: 10.1029/2024MS004292](https://doi.org/10.1029/2024MS004292)
* [Convection parameterisations in ICON](https://github.com/EyringMLClimateGroup/heuer23_ml_convection_parameterization) -
  Implementing machine-learnt convection parameterisations in the ICON atmospheric model
  showing that best online performance occurs when causal relations are eliminated from the net.
  See Heuer et al (2024) - [DOI: 10.1029/2024MS004398](https://doi.org/10.1029/2024MS004398)
* In the [GloSea6 Seasonal Forecasting Model](https://www.metoffice.gov.uk/research/climate/seasonal-to-decadal/gpc-outlooks/user-guide/global-seasonal-forecasting-system-glosea6) -
  Replacing a BiCGStab bottleneck in the code with a deep learning approach to speed up execution without compromising model accuracy.
  See Park and Chung (2025) - [DOI: 10.3390/atmos16010060](https://doi.org/10.3390/atmos16010060)
