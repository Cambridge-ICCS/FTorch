# Example 5 - Looping

So far many of the examples have been somewhat trivial, reading in a net and calling it
once to demonstrate the inference process.

In reality most applications that we use Fortran for will be performing many iterations
of a process and calling the net multiple times.
Loading in the net from file is computationally expensive, so we should do this only
once, and then call the forward method on the loaded net as part of the iterative
process.

This example demonstrates the naive 'bad' approach and then the more efficient 'good'
approach. It shows the suggested way to break down the FTorch code into initialisation,
forward, and finalisation subprocesses, and allows users to time the different
approaches to observe the significant performance difference.

## Description

We revisit SimpleNet from the first example that takes an input tensor of length 5
and multiplies it by two.
This time we start by passing it the the tensor `[1.0, 2.0, 3.0, 4.0]`, but then iterate
10,000 times, each time incrementing each element by 1.0.
We sum the results of each forward pass and print the final result.

There are two folders `bad/` and `good/` that show two different approaches.

The same `pt2ts.py` tool as in the previous examples is used to save the
network to TorchScript. A `simplenet_infer_fortran.f90` file contains the main
program that runs over the loop. A `fortran_ml_mod.f90` file contains a module with
the FTorch code to load the TorchScript model, run it in inference mode, and clean up.

