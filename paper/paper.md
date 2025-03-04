---
title: 'FTorch: a library for coupling PyTorch models to Fortran'
tags:
  - Fortran
  - Python
  - PyTorch
  - machine learning
authors:
  - name: Jack Atkinson
    orcid: 0000-0001-5001-4812
    affiliation: "1"
    corresponding: true
  - name: Athena Elafrou
    affiliation: "2"
  - name: Elliott Kasoar
    affiliation: "1, 3"
    orcid: 0009-0005-2015-9478
  - name: Joseph G. Wallwork
    affiliation: "1"
    orcid: 0000-0002-3646-091X
  - name: Thomas Meltzer
    affiliation: "1"
    orcid: 0000-0003-1740-9550
  - name: Simon Clifford
    affiliation: "1"
    orcid: 0000-0001-7754-504X
  - name: Dominic Orchard
    affiliation: "1, 4"
    orcid: 0000-0002-7058-7842
  - name: Chris Edsall
    affiliation: "1"
    orcid: 0000-0001-6863-2184
affiliations:
 - name: Institute of Computing for Climate Science, University of Cambridge, UK
   index: 1
 - name: NVIDIA, UK
   index: 2
 - name: Scientific Computing Department, Science and Technology Facilities Council, UK
   index: 3
 - name: University of Kent, UK
   index: 4
date: 25 March 2024
bibliography: paper.bib

---

# Summary

In the last decade, machine learning (ML) and deep learning (DL) techniques have
revolutionised many fields within science, industry, and beyond.
Researchers across domains are increasingly seeking to combine ML
with numerical modelling to advance research.
This typically brings about the challenge of _programming
language interoperation_. PyTorch [@paszke2019pytorch] is a popular framework for
designing and training ML/DL models whilst Fortran remains a language of choice for many
high-performance computing (HPC) scientific models.
The `FTorch` library provides an easy-to-use, performant, cross-platform method for
coupling the two, allowing users to call PyTorch models from Fortran.

`FTorch` is open-source, open-development, and well-documented with minimal dependencies.
A central tenet of its design, in contrast to other approaches, is
that FTorch removes dependence on the Python runtime (and virtual environments).
By building on the `LibTorch` backend (written in C++ and accessible via an API) it
allows users to run ML models on both
CPU and GPU architectures without needing to port code to device-specific languages.


# Statement of need

The explosion of ML/DL has brought several promising opportunities
to deploy these techniques in scientific research.
There are notable applications in the physical sciences [@carleo2019machine],
climate science [@kashinath2021physics],
and materials science [@bishara2023state].
Common applications include the emulation of computationally intensive processes
and the development of data-driven components.
Such deployments of ML can achieve improved computational and/or predictive performance,
compared to traditional numerical techniques.
A common example from the geosciences is ML parameterisation
of subgrid processes &mdash; a major source of uncertainty in many models
[e.g. @bony2015clouds; @rasp2018deep].

Fortran is widely used for scientific codes due to its performance,
stability, array-oriented design, and native support for shared and distributed memory,
amongst other features [@kedward2022state].
Many ML frameworks, on the other hand, are accessed using Python.
The commonly-used PyTorch framework
allows users to design and deploy ML models with many advanced features.

Ideally, users would develop and validate ML models in the PyTorch environment
before deploying them into a scientific model.
This deployment should require minimal additional code, and guarantee
identical results as obtained with the PyTorch
interface &mdash; something not guaranteed if re-implementing by hand in Fortran.
Ideally one would call out, from Fortran, to an ML model
saved from PyTorch, with the results returned directly to the scientific code.

`FTorch` bridges this gap, reducing the burden on researchers
seeking to incorporate ML into their numerical models.
It provides an intuitive and user-friendly interface from Fortran
to ML models developed using PyTorch.
It removes the need for detailed knowledge about language interoperation
and the need to develop bespoke coupling code, instead providing a Fortran interface
designed to be familiar to both PyTorch and Fortran users.

Having no dependence on the Python runtime, `FTorch` avoids the
incurred overhead of object representations in Python's runtime and
is appealing for HPC systems where the management of
Python environments can be challenging.


# Software description

`FTorch` is a Fortran wrapper to the `LibTorch` C++ framework using the `iso_c_binding`
module, intrinsic to Fortran since the 2003 standard
This enables shared memory use (where possible) to
maximise efficiency by reducing data-transfer during coupling^[i.e. the same
data in memory is used by both `LibTorch` and Fortran without creating a copy.]
and avoids any use of Python at runtime.
PyTorch types are represented through derived types in `FTorch`, with Tensors supported
across a range of data types and ranks using the `fypp` preprocessor [@fypp].

We utilise the existing support in `LibTorch` for
GPU acceleration without additional device-specific code.
`torch_tensor`s are targeted to a device through a
`device_type` enum, currently supporting CPU, CUDA, XPU, and MPS.
Multiple GPUs may be targeted through the optional `device_index` argument.

Typically, users train a model in PyTorch and save it as TorchScript, a strongly-typed
subset of Python.
Once saved, the Torchscript model can be loaded from Fortran using `FTorch` and run
via the `LibTorch` backend.
The library comes with a utility script (`pt2ts.py`) to assist with the process of
saving models as well as a comprehensive set of examples guiding users
through complete Python to Fortran workflows.
A focus on user experience underpins the development and is a key aspect behind the
adoption of `FTorch` by various scientific communities.

Full details, including user guide, API documentation, slides and videos, and links to
projects is available at 
[https://cambridge-iccs.github.io/FTorch](https://cambridge-iccs.github.io/FTorch).


# Comparison to other approaches

* **Replicating a net in Fortran**\
  That is, taking a model developed in PyTorch and reimplementing
  it using only Fortran, loading saved weights from file.
  This is likely to require considerable development effort, re-writing code
  that already exists and missing opportunities to use the diverse and highly-optimised
  features of Torch. Re-implementation can be a source of
  error, requiring additional testing to ensure correctness.\
  If the overall goal is to incorporate ML into Fortran, rather than using PyTorch
  specifically, then another approach is to leverage a Fortran-based ML framework
  such as [neural-fortran](https://github.com/modern-fortran/neural-fortran) [@curcic2019parallel].
  Whilst it does not allow interaction with PyTorch, neural-fortran provides
  many neural network components for building nets directly in Fortran.
  However, the set of features is not as rich as PyTorch and GPU offloading
  is not currently supported.\
  The [Fiats](https://berkeleylab.github.io/fiats) (`Functional Inference And Training
  for Surrogates') library [@fiats] is another approach for developing, training, and deploying
  ML models directly in Fortran, with experimental GPU support at present.

* **Forpy** [@forpy]\
  Forpy is a Fortran module that provides access to Python data structures (including `numpy` arrays)
  for interoperability.
  Whilst this provides wider access to general Python features it has a
  challenging interface with more boilerplate. It also requires access to the Python
  runtime from Fortran.

* **TorchFort** [@torchfort]\
  Since we began `FTorch` NVIDIA has released `TorchFort`.
  This has a similar approach to `FTorch`, avoiding Python to link against
  the `LibTorch` backend. It has a focus on enabling GPU deployment on NVIDIA hardware.

* **fortran-tf-lib** [@fortran-tf-lib]\
  Whilst `FTorch` allows coupling of PyTorch models to Fortran,
  some use TensorFlow [@Abadi_TensorFlow_Large-scale_machine_2015] to
  develop and train models. These can be coupled to
  Fortran in a similar manner to `FTorch` by using `fortran-tf-lib`.
  Whilst it provides an alternative solution, the library is less rich in features and
  software sustainability than `FTorch`.

* **SmartSim** [@partee2022using]\
  SmartSim is a workflow library developed by HPE and built upon Redis API.
  It provides a framework for launching ML and HPC workloads transferring data
  between the two via a database.
  This is a versatile approach that can work with a variety of languages and ML
  frameworks. However, it has a significant learning curve, incurs data-transfer
  overheads, and requires managing tasks from Python.

# Examples of Use

`FTorch` is actively used in scientific research:

* in the [DataWave project](https://datawaveproject.github.io/) exploring gravity wave
  drag in atmospheric models to:

  * couple an emulator into the MiMA model [@MiMAML] demonstrating variability of models
    trained offline when coupled back to a host [@mansfield2024uncertainty].

  * couple emulators and new data-driven parameterisations to the Community Atmosphere
    Model (CAM).

* to couple a U-Net based model of multi-scale convection into [ICON](https://www.icon-model.org/) [@ICON]
  and demonstrate via Shapley values that
  non-causal learnt relations are more stable when running online [@heuer2024interpretable].

* As part of [CESM](https://www.cesm.ucar.edu/) (the Community Earth System Model) [@CESM]
  working to provide a general approach for researchers to couple ML models to the
  various components of the model suite.


# Future development

Recent work in scientific domains suggests that online training is
likely important for long-term stability of hybrid models [@brenowitz2020machine].
We therefore plan to extend FTorch to expose PyTorch's autograd functionality to support this.

We welcome feature requests and are open to discussion and collaboration.


# Acknowledgments

This project is supported by Schmidt Sciences, LLC. We also thank
the Institute of Computing for Climate Science for their support.


# References
