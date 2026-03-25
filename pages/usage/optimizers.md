title: Optimizers API
author: Jack Atkinson
date: Last Updated: March 2026

## Optimizers API Documentation

- [Overview](#overview)
- [Procedures](#procedures)
    - [Constructors](#constructors)
    - [Core Methods](#core-methods)
    - [Deallocation](#deallocation)
    - [Usage](#usage)

### Overview

FTorch provides an [[ftorch_optim(module):torch_optim(type)]] derived type exposing the
functionality of the `torch::optim` C++ class.
The interface is designed to be familiar to Fortran programmers, whilst retaining strong
similarity with `torch::optim` class and the `torch.optim` Python package.
This includes default values for optional tuning parameters.

This module enables Fortran programmers to use familiar optimizers like SGD, Adam,
and AdamW through a Fortran-friendly interface to perform optimization steps to tensors.

The [[ftorch_optim(module):torch_optim(type)]] type holds a pointer to a PyTorch
optimizer object in C++ (implemented using `c_ptr` from the `iso_c_binding` intrinsic module).
This avoids unnecessary data copies and provides direct access to Torch's
optimization capabilities.


### Procedures

#### Constructors

FTorch currently provides three optimizer constructors, each corresponding to a
popular PyTorch optimization algorithm.
All of these are created by a call to a specific subroutine that takes as inputs
a [[ftorch_optim(module):torch_optim(type)]] type to assign
the created optimizer to, and an array of [[ftorch_tensor(module):torch_tensor(type)]]
objects to optimize.

In addition they can each take a number of optional tuning parameters with defaults set
to match those of PyTorch. Details of these appear in the API pages linked below.

- **SGD** (Stochastic Gradient Descent) is implemented through
  [[ftorch_optim(module):torch_optim_SGD(subroutine)]]
- **Adam** (Adaptive Moment Estimation) is implemented through
  [[ftorch_optim(module):torch_optim_Adam(subroutine)]]
- **AdamW** (Decoupled Weight Decay Adam) is implemented through
  [[ftorch_optim(module):torch_optim_AdamW(subroutine)]]

#### Core Methods

Whilst different subroutines exist for creating different kinds of optimizer, they all
have common core methods that are used.

##### Zero Gradients

[[ftorch_optim(module):torch_optim_zero_grad(subroutine)]] clears the gradients of all
parameters managed by the optimizer. This should be called at the beginning of each
iteration during training.
The method is implemented as a procedure bound to the
[[ftorch_optim(module):torch_optim(type)]] type and can be called as:
`optimizer%zero_grad()`.

##### Step

[[ftorch_optim(module):torch_optim_step(subroutine)]] performs a single optimization
step, updating all parameters managed by the optimizer based on their gradients.
It should be called after backpropogation has been performed following a forward pass
during a training iteration.

The method is implemented as a procedure bound to the
[[ftorch_optim(module):torch_optim(type)]] type and can be called as:
 `optimizer%step()`.

#### Deallocation

[[ftorch_optim(module):torch_optim_delete(subroutine)]] deallocates the memory
associated with an optimizer.
It is implemented as the finalizer of the [[ftorch_optim(module):torch_optim(type)]]
type so will automatically be called when the optimizer goes out of scope.
See the [Fortran-lang page on object-oriented Fortran](https://fortran-lang.org/learn/oop_features_in_fortran/object_based_programming_techniques/#finalization-and-conclusions)
for further details about finalization.

#### Usage

The typical usage pattern for FTorch optimizers follows the standard PyTorch training loop:

```fortran
type(torch_tensor) :: tensor, output, target_data, loss
type(torch_optim) :: optimizer

! Create optimizer - here we use SGD
call torch_optim_SGD(optimizer, [tensor], learning_rate=0.01D0)

! Training loop
do i = 1, n_epochs

  ! Zero gradients
  call optimizer%zero_grad()
  
  ! Forward pass and loss calculation
  call my_forward_pass(tensor, output)
  call torch_tensor_mean(loss, (output - target_data) ** 2)
  
  ! Backward pass
  call torch_tensor_backward(loss)
  
  ! Optimization step
  call optimizer%step()
  
end do
```

For more details on backpropogation and autograd, and use of
optimizers as part of the training process, see the
[online training](|page|/usage/online.html) documentation.

@note
For a concrete example of how to use the various optimizer methods as part of a
training loop see the
[optimizers worked example](|page|/usage/worked_examples.html).
@endnote
