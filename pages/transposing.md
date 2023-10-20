title: When to transpose data

In the ResNet18 example, it was expected that the shape and indices of `in_data` in resnet_infer_fortran.f90 match that of `input_batch` in resnet18.py, i.e. `in_data(i, j, k, l) ==  input_batch[i, j, k, l]`.

Since C is row-major (rows are contiguous in memory), whereas Fortran is column-major (columns are contiguous), it is therefore necessary to perform a transpose when converting from the NumPy array to the Fortran array to ensure that their indices are consistent.

In this example code, the NumPy array is transposed before being flattened and saved to binary, allowing Fortran to `reshape` the flatted array into the correct order.

An alternative would be to save the NumPy array with its original shape, but perform a transpose during or after reading the data into Fortran, e.g. using:

```
in_data = reshape(flat_data, shape(in_data), order=(4,3,2,1))
```

For more general use, it should be noted that the function used to create the input tensor from `input_batch`, `torch_tensor_from_blob`, performs a further transpose, which is required to allow the tensor to interact correctly with the model.
