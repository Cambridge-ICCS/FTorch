# Example 13 - GPU device pointer

This example explores whether it is possible to construct a `torch_tensor`
directly on the GPU by passing a GPU device pointer to `torch_tensor_from_blob`,
rather than creating tensors on the CPU and moving them to the GPU with
`torch_tensor_to` (or equivalently specifying a GPU device in one of the
standard constructors such as `torch_tensor_ones`).

## Description

A CUDA Fortran file `gpu_tensor_pointer.f90` is provided that:

* allocates an array on the GPU using CUDA Fortran and obtains a device pointer
  to it with `c_devloc`;
* constructs a `torch_tensor` from the device pointer using
  `torch_tensor_from_blob`, supplying shape and stride information explicitly
  (since `torch_tensor_from_array` cannot be used with CUDA Fortran `device`
  arrays);
* interrogates the resulting tensor and prints its contents to determine
  whether the device pointer has been respected;
* performs a quantitative check that GPU arithmetic on the tensor produces the
  expected values in GPU memory, by assigning the result into an output tensor
  constructed from another GPU device pointer and comparing with a result
  computed in pure Fortran.

When inspecting `torch_from_blob` in `src/ctorch.cpp` it will be seen that
`torch::from_blob` is called with CPU `TensorOptions`, with the GPU device only
applied afterwards via `.to(device)`. It is therefore not obvious a priori
whether a device pointer is respected (libtorch infers the device of the data
from its placement in memory) or whether host memory is assumed (which would
lead to a crash or garbage values). Compiling and running this example answers
that question.

## Dependencies

To run this example requires:

- CMake
- An NVIDIA GPU
- A CUDA Fortran capable compiler (e.g. `nvfortran`) - required for the
  `cudafor` module, `device` attribute, and `c_devloc` procedure
- FTorch installed as described in the main package, **built with
  `-DGPU_DEVICE=CUDA`**

## Running

To compile the Fortran code, using the included `CMakeLists.txt`, execute the
following:

```
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=<path/to/your/installation/of/library/> \
         -DCMAKE_Fortran_COMPILER=nvfortran \
         -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

Then run the compiled executable:

```
./gpu_tensor_pointer
```

You will also likely need to add the location of the dynamic library files
(`.so` files) to your `LD_LIBRARY_PATH`:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:</path/to/library/installation>/lib
```

## Interpreting the output

The program prints the values it interrogates rather than just pass/fail so the
behaviour can be interpreted:

* If the example runs to completion with the expected device type
  (`torch_kCUDA`), values, and allclose check passing, then
  `torch_tensor_from_blob` can wrap GPU memory directly, avoiding a
  host-to-device copy.
* If the program crashes, or the tensor prints garbage values, then the
  `torch::from_blob` wrapper in FTorch assumes host memory and device pointers
  cannot currently be passed through this route - in which case tensors must
  continue to be created on the CPU and moved with `torch_tensor_to`.
