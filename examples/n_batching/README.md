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
- **Shape:** The last dimension of your input must match the model's expected input
  size (here, 5) otherwise it will result in an error.
- **Location of Batching Dimension:** PyTorch modules such as `nn.Linear` always operate
  on the last dimension of the input tensor. Any number of leading (batching) dimensions
  are supported, and the operation is applied independently across all of them.
  For example, `[batch, 5]`, `[batch, time, 5]`, or even `[batch1, batch2, ..., 5]` are
  all valid as long as the final dimension is consistent with the feature dimension.
- **Contiguity:** If you slice or permute tensors, ensure they are contiguous in memory
  before passing to the model (use `.contiguous()` if needed).


## Running
1. (Recommended) Create a virtual environment and install dependencies:
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run the demo script:
   ```
   python3 batching_demo.py
   ```

The output should match the results described above in the "Why Batching?" section as
well as an example of batching failing due to the batching dimension being at the wrong
index.

---

## Further Exploration
- Try using different batch and time dimensions.
- Experiment with slicing and permuting tensors before inference.
- Extend the model to handle different input sizes or more complex architectures.

