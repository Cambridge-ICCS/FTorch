title: Usage
author: Jack Atkinson
date: Last Updated: October 2025
ordered_subpage: generic_example.md
ordered_subpage: worked_examples.md
ordered_subpage: tensor.md
ordered_subpage: transposing.md
ordered_subpage: batching.md
ordered_subpage: offline.md
ordered_subpage: online.md
ordered_subpage: troubleshooting.md


## Usage

- [Examples](#examples)
    - [Generic Example](|page|/usage/generic_example.html)
    - [Worked Examples](|page|/usage/worked_examples.html)
- [API Documentation](#api-documentation)
    - [Tensor API](|page|/usage/tensor.html)
        - [Transposing data](|page|/usage/transposing.html)
    - Model API (WIP)
        - [Batching inference](|page|/usage/batching.html)
    - Optimizers API (WIP)
- [Training](#training)
    - [Offline](|page|/usage/offline.html)
    - [Online](|page|/usage/online.html)
- [Troubleshooting](|page|/installation/troubleshooting.html)


### Examples

The [Overview & Generic Example](|page|/usage/generic_example.html) page walks
through the process of saving a model from Python and using it within Fortran,
including how to build and link the code using FTorch to the library.
This is the best place to start to understand how FTorch works and how to use it.

The [Worked Examples](|page|/usage/worked_examples.html) page summarises the
comprehensive set or practical examples included in the FTorch repository.
These demonstrate how to use FTorch for model coupling, tensor manipulation, with GPU
acceleration, with MPI, and more.
It is advised to work through some of these to check your installation and better
understand how to use FTorch.


### API Documentation

These pages contain detailed documentation of the various component APIs included in FTorch.
Currently there is detail for [Tensors](|page|/usage/tensor.html), with Models and
Optimizers being work in progress.
There is also a page explaining how [batched inference](|page|/usage/batching.html)
works in FTorch, including key tips.


### Training

These pages discuss the conceptual ideas around using FTorch for coupling models
trained [offline](|page|/usage/offline.html), and training models
[online](|page|/usage/online.html).
