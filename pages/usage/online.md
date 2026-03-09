title: Online training
author: Joe Wallwork
date: Last Updated: March 2026

## Online Training

FTorch has supported offline training of ML models for some time (see the
[offline training user guide page](|page|/usage/offline.html) for details). We are
currently working on extending its functionality to support online training
by exposing the backpropagation and optimization
functionalities of PyTorch/LibTorch.

Below we provide a schematic of the online training workflow, which is broken
down into separate tasks below.

![schematic](|media|/usage/online.svg "Online training schematic")

To set up online training, you will need to make use of the backpropagation and
optimization functionalities of PyTorch/LibTorch, which have been exposed in
FTorch. Details of how to do this are provided in the following.

#### 1. Design the ML model in PyTorch

This task is identical to the offline case. It is done purely in Python and is
not described here. See the
[PyTorch documentation](https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html)
for information on how to do this.

#### 2. pt2ts

The scripting section comes earlier in the online workflow. Having written a
model to a file with `.pt` extension, use the `pt2ts.py` utility Python script
to convert it to TorchScript format. A template `pt2ts.py` script can be found
in the [`utils`](https://github.com/Cambridge-ICCS/FTorch/tree/main/utils)
subdirectory. See the
[README](https://github.com/Cambridge-ICCS/FTorch/blob/main/utils/README.md)
there for more details on how to use the script.

#### 3. Data generation and training

In the online case, data generation and training are done in the same step,
purely in Fortran. Code modifications are made so that we can run the Fortran
model to generate training data and immediately use this training data (whilst
in memory) to train the ML model. There is not necessarily an optimization loop
in this case - one option is to take a pre-trained model and to continue
improving it using data generated online.

The training code should be set up such that the file containing the TorchScript
model that was created in step 2 is read in at the start of the Fortran program
and the modified ML model is written out to the same TorchScript file at the
end of the Fortran program. This way, the model can be trained in multiple
Fortran runs, with the same model being read.

*Note: Training and writing out models has not yet been implemented in FTorch,
but is work in progress.*

#### 4. Fortran model with inference

In order to run inference with the trained ML model, you will need to create
another modified version of your Fortran model that loads the TorchScript model
and uses FTorch syntax to set up appropriate `torch_tensor` and `torch_model`
objects and call the [[ftorch_model(module):torch_model_forward(subroutine)]]
subroutine to run the inference. For
examples of how to do this, see the
[optimizer worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/n_Optimizers).

### Defining as much as possible in the model

The first thing to note is that it's best to define as much as possible in the
PyTorch model before writing it to file. As well as layers, activation
functions, and loss functions, PyTorch models can also contain expressions
involving tensors, e.g., mathematical operations. If you intend to include such
expressions in your code then it is best to do this in the model definition, if
possible. This ensures that such operations are handled by LibTorch, meaning
there can be no overheads related to the coupling with Fortran.

If you have developed a custom loss function, for example, see if you can define
it in PyTorch. Some functionality for handling tensor operations has been
exposed in FTorch - as detailed below - but you will have the most functionality
available to you if you write such code into your PyTorch model.

Reasons that it might not be possible to write all of your operations into your
PyTorch model include scripting errors for certain operations and advanced
custom loss functions that involve downstream Fortran code.

### Operator overloading

Mathematical operators involving Tensors are overloaded, so that we can compute
expressions involving outputs from one or more ML models. For more information
on this, see the [tensor API](|page|/usage/tensor.html) documentation page.

Whilst it's possible to import such functionality with a bare
```fortran
use ftorch
```
statement, the best practice is to import specifically the operators that you
wish to use. Note that the assignment operator `=` has a slightly different
notation:
```fortran
use ftorch, only: assignment(=), operator(+), operator(-), operator(*), &
  operator(/), operator(**)
```

If you would like to make use of scalar multiplication or scalar division, this
can be achieved by setting the scalar as a rank-1
[[ftorch_tensor(module):torch_tensor(type)]] with a single entry. For example:
```fortran
multiplier_array(1) = 3.0_wp
call torch_tensor_from_array(multiplier, multiplier_value, torch_kCPU)
```

For a concrete example of how to compute mathematical expressions involving
Torch tensors, see the
[autograd worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/8_Autograd).


### The `requires_grad` property

For Tensors that you would like to differentiate with respect to, be sure to
set the `requires_grad` optional argument to `.true.` when you construct it.


### Backpropagation

Having defined some tensors with the `requires_grad` property set to `.true.`
and computed another tensor in terms of an expression involving these, we can
compute gradients of that tensor with respect to those that it depends on. This
is achieved using the
[[ftorch_tensor(module):torch_tensor_backward(subroutine)]] subroutine. For
example, for input tensors `a` and `b` and an output tensor `Q`:

```fortran
call torch_tensor_from_array(a, in_data1, tensor_layout, torch_kCPU, &
                             requires_grad=.true.)
call torch_tensor_from_array(b, in_data2, tensor_layout, torch_kCPU, &
                             requires_grad=.true.)
call torch_tensor_from_array(Q, out_data1, tensor_layout, torch_kCPU)

Q = a * b

call torch_tensor_backward(Q)
```

Following the example code above, we can extract gradients of `Q` with respect
to `a` and/or `b`. To do this, we can use the
[[ftorch_tensor(module):torch_tensor_get_gradient(subroutine)]] subroutine. That
is, for tensors `dQda` and `dQdb`:

```fortran
call torch_tensor_from_array(dQda, out_data2, tensor_layout, torch_kCPU)
call torch_tensor_get_gradient(dQda, a)

call torch_tensor_from_array(dQdb, out_data3, tensor_layout, torch_kCPU)
call torch_tensor_get_gradient(dQdb, b)
```

#### `retain_graph` argument

If you wish to call the backpropagation operator multiple times then you may
need to make use of the `retain_graph` argument for
[[ftorch_tensor(module):torch_tensor_backward(subroutine)]]. This argument
accepts logical values and defaults to `.false.`, for consistency with PyTorch
and LibTorch. According to the
[PyTorch docs](https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html),
`retain_graph=.true.` will not be needed in most cases, but it's useful to have
for the cases where it is.

#### Zeroing gradients

Having computed gradients of one tensor with respect to its dependencies,
suppose you wish to compute gradients of another tensor. Since the gradient
values associated with each dependency are accumulated, you should zero the
gradients before computing the next gradient. This can be achieved using the
[[ftorch_tensor(module):torch_tensor_zero_grad(subroutine)]] subroutine.

Following the example code above:

```fortran
Q = a * b
P = a + b

call torch_tensor_backward(Q)

! ...

call torch_tensor_zero_grad(a)
call torch_tensor_zero_grad(b)

call torch_tensor_backward(P, retain_graph=.true.)

! ...
```

#### Extracting gradients

Note that [[ftorch_tensor(module):torch_tensor_get_gradient(subroutine)]] must
be called after every call to
[[ftorch_tensor(module):torch_tensor_backward(subroutine)]] or
[[ftorch_tensor(module):torch_tensor_zero_grad(subroutine)]], even if the
gradient for the same tensor is being extracted into the same array. This is due
to the way that pointers are handled on the C++ side.


### Optimization

FTorch now supports running optimizers. That is, it's possible to do training in
Fortran as well as in Python. To make use of optimizers in FTorch, you need the
`torch_optim` derived type. This derived type has two member subroutines as
follows:

* `torch_optim%zero_grad` ([[ftorch_optim(module):torch_optim_zero_grad(subroutine)]]),
  which zeroes all tensors associated with the
  optimizer. This should be called at the beginning of every step of the
  optimization loop.
* `torch_optim%step` ([[ftorch_optim(module):torch_optim_step(subroutine)]]),
  which takes an iteration of the optimization method.

The [optimizer worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/n_Optimizers)
is probably the best place to get started to see how to use the functionality.


### Loss functions

The loss function classes defined in PyTorch/LibTorch have not yet been exposed
in FTorch. However, the [[ftorch_tensor(module):torch_tensor_sum(subroutine)]] and
[[ftorch_tensor(module):torch_tensor_mean(subroutine)]] reduction
operators have been provided, which should be sufficient for simple loss
functions such as the mean-square-error (MSE).

See the [optimizer worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/n_Optimizers)
for an example of how to use [[ftorch_tensor(module):torch_tensor_mean(subroutine)]]
to define a MSE loss function.
