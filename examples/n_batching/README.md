# Example: Batching and Higher-Dimensional Inference with PyTorch

This example demonstrates how to use a PyTorch model trained on 1D vectors to perform
inference on batched and higher-dimensional data. We start with a simple batch and then
show how to handle more complex cases, such as time series or multi-dimensional data.

## Description
A Python file `batchingnet.py` is provided that defines a simple PyTorch 'BatchingNet'
that takes an input vector of length 5 and applies a single `Linear` layer to multiply
each input feature by a different value (0, 1, 2, 3, 4). The demo script
`batching_demo.py` shows how to use this model for unbatched, batched, and
higher-dimensional inference, illustrating the effect of batching and the location of
the batch dimension. All outputs are as described in the "Why Batching?" section below.


## Dependencies
To run this example requires:

- CMake
- Fortran compiler
- FTorch (installed as described in main package)
- Python 3


## Why Batching?
Batching allows you to efficiently process multiple inputs at once, leveraging vectorised
operations and hardware acceleration. PyTorch modules are designed to handle batched
inputs by default, as long as the input's last dimension matches the expected feature size.
The outputs for each case are shown below so you can check your results.

### Unbatched Inference
Suppose you have a model trained on input vectors of shape `[5]`. You can run inference
on a single vector as follows:

```
input_tensor = torch.ones(5)  # shape: [5]
output = model(input_tensor)
```
Input:
```
tensor([1., 1., 1., 1., 1.])
```
Expected output:
```
tensor([0., 1., 2., 3., 4.])
```


### Simple Batch Inference
You can run inference on a batch of such vectors by stacking them into a tensor of
shape `[batch_size, 5]`.

```
input_tensor = torch.stack([
    torch.ones(5),
    2 * torch.ones(5),
])
output = model(input_tensor)
```
Input:
```
tensor([[1., 1., 1., 1., 1.],
        [2., 2., 2., 2., 2.]])
```
Expected output:
```
tensor([[0., 1., 2., 3., 4.],
        [0., 2., 4., 6., 8.]])
```

The model will process each row independently.


### Higher-Dimensional Inference (e.g., [batch, time, 5])
You can also run inference on data with more dimensions, such as `[batch, time, 5]`.
PyTorch modules like `nn.Linear` operate on the last dimension, so this works as long
as the last dimension matches the model's input size.

For example, using a tensor with shape `[2, 3, 5]`:

```
input_tensor = torch.stack([
    torch.stack([
        torch.ones(5),
        2 * torch.ones(5),
        3 * torch.ones(5),
    ]),
    torch.stack([
        10 * torch.ones(5),
        20 * torch.ones(5),
        30 * torch.ones(5),
    ]),
])
output = model(input_tensor)
```
Input:
```
tensor([[[ 1.,  1.,  1.,  1.,  1.],
         [ 2.,  2.,  2.,  2.,  2.],
         [ 3.,  3.,  3.,  3.,  3.]],

        [[10., 10., 10., 10., 10.],
         [20., 20., 20., 20., 20.],
         [30., 30., 30., 30., 30.]]])
```
Expected output:
```
tensor([[[  0.,   1.,   2.,   3.,   4.],
         [  0.,   2.,   4.,   6.,   8.],
         [  0.,   3.,   6.,   9.,  12.]],

        [[  0.,  10.,  20.,  30.,  40.],
         [  0.,  20.,  40.,  60.,  80.],
         [  0.,  30.,  60.,  90., 120.]]])
```



## Key Considerations
- **Shape:** The last dimension of the input must match the model's expected input
  size (here, 5) otherwise it will result in an error.
- **Location of Batching Dimension:** PyTorch modules such as `nn.Linear` always operate
  on the last dimension of the input tensor. Any number of leading (batching) dimensions
  are supported, and the operation is applied independently across all of them.
  For example, `[batch, 5]`, `[batch, time, 5]`, or even `[batch1, batch2, ..., 5]` are
  all valid as long as the final dimension is consistent with the feature dimension.
- **Contiguity:** If you slice or permute tensors, ensure they are contiguous in memory
  before passing to the model (use `.contiguous()` if needed).


## Running

To run this example, first install FTorch as described in the main documentation. Then, from this directory, create a virtual environment and install the necessary Python modules:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You can see how you would perform batching in PyTorch by running the Python batching
demo:
```
python3 batching_demo.py
```
This will run the BatchingNet model with various inputs and print the expected outputs
as described above.

To save the BatchingNet model to TorchScript for use in Fortran, run:
```
python3 pt2ts.py
```
This will generate `saved_batchingnet_model_cpu.pt` in the current directory.

At this point you no longer require Python, so can deactivate the virtual environment:
```
deactivate
```

To call the saved BatchingNet model from Fortran, repeating the different batching
approaches in the Python demo compile the `batchingnet_infer_fortran.f90` file.
This can be done using the included `CMakeLists.txt` as follows:
```
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=<path/to/your/installation/of/library/> -DCMAKE_BUILD_TYPE=Release
cmake --build .
```
(Note that the Fortran compiler can be chosen explicitly with the `-DCMAKE_Fortran_COMPILER` flag, and should match the compiler that was used to locally build FTorch.)

To run the compiled code calling the saved BatchingNet TorchScript from Fortran, run the executable with an argument of the saved model file:
```
./batchingnet_infer_fortran ../saved_batchingnet_model_cpu.pt
```

This runs the model with single, batched, and multidimensional batched input arrays and should produce the output:
```
--- Single input output: ---
0.00000000       1.00000000       2.00000000       3.00000000       4.00000000

---Batched input output: ---
   0.00000000       1.00000000       2.00000000       3.00000000       4.00000000
   0.00000000       2.00000000       4.00000000       6.00000000       8.00000000

--- Multidimensional batched input/output: ---
Input (1,1):    1.00000000       1.00000000       1.00000000       1.00000000       1.00000000
Output (1,1):   0.00000000       1.00000000       2.00000000       3.00000000       4.00000000
Input (2,3):    30.0000000       30.0000000       30.0000000       30.0000000       30.0000000
Output (2,3):   0.00000000       30.0000000       60.0000000       90.0000000       120.000000

BatchingNet Fortran example ran successfully
```

---

## Further Exploration

- incorrect Dimension order
    - A commented-out line in the Fortran code demonstrates how an incorrect feature
      dimension (e.g., `[2, 4]`) will cause a runtime error.
      ```fortran
      call torch_model_forward(model, in_tensors_bad, out_tensors_bad)
      ```
    - Uncomment, rebuild and re-run to observe the error message for incorrect
      batching layout.
- Try using different batch and time dimensions.
- Change the nature of the model and observe the effect of batching.
