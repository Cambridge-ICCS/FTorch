# Example 10 - Optimizers

This example demonstrates the use of optimizers in FTorch by
leveraging PyTorch's optim module.

By exposing optimizers in Fortran, FTorch is able to compute optimisation
steps to update tensors and models as part of a training process.

## Description

A Python demo that trains a single tensor to map an input vector to a
target is provided as `optimizers.py`, showing how to use an optimiser in PyTorch.

The demo is then replicated in Fortran as `optimizers.f90`, to show how
the same process and result can be achieved using FTorch.

## Dependencies

To run this example requires:

- CMake
- Fortran compiler
- FTorch (installed as described in main package)
- Python 3

## Running

To run this example, first install FTorch as described in the main
documentation, making use of the `examples` dependency group.

Run the Python version of the demo with
```
python3 optimizers.py
```
This trains a tensor to scale, elementwise, a vector of ones to the vector `[1, 2, 3, 4]`.
It uses the torch SGD optimiser to adjust the values of the scaling tensor at each step,
outputting values of interest to screen in the form:
```console
========================
Epoch: 0
        Output:
                tensor([1., 1., 1., 1.], grad_fn=<MulBackward0>)
        loss:
                3.5
        tensor gradient:
                tensor([ 0.0000, -0.5000, -1.0000, -1.5000])
        tensor:
                tensor([1.0000, 1.5000, 2.0000, 2.5000], requires_grad=True)
...
```

To run the Fortran version of the demo we need to compile with (for example)
```
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=<path/to/your/installation/of/library/> -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

(Note that the Fortran compiler can be chosen explicitly with the `-DCMAKE_Fortran_COMPILER` flag,
and should match the compiler that was used to locally build FTorch.)

To run the compiled code, simply use
```
./optimizers
```
Again, this uses the SGD optimiser to adjust the values of the scaling tensor at
each step, outputting values of interest to screen in the form:
```console
 ================================================
 Epoch:            1

 Output:   1.00000000       1.00000000       1.00000000       1.00000000

 loss:
 3.5000
[ CPUFloatType{1} ]

 tensor gradient:
 0.0000
-0.5000
-1.0000
-1.5000
[ CPUFloatType{4} ]

 scaling_tensor:
 1.0000
 1.5000
 2.0000
 2.5000
[ CPUFloatType{4} ]
...
```

## Plotting the loss function convergence and checking for consistency

The Python and Fortran versions of the demo both output the values
of the loss function at each epoch to files `pytorch_losses.dat` and
`ftorch_losses.dat`, respectively. These can be compared to check that
they match numerically and plotted using the `plot_and_compare.py`
script which can be run with:
```sh
python3 plot_and_compare.py
```
which will read the data files and produce a `losses.png` file in the current
directory. The result should look something like

![Losses](expected_losses.png)

That is, the loss function convergence behaviour is the same with PyTorch and
FTorch, as we might expect.
