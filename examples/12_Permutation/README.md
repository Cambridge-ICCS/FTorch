# Example 12 - Tensor permutation

This example demonstrates how to permute tensor dimensions when constructing
tensors from Fortran arrays, and how Fortran's column-major memory layout
relates to Torch's row-major layout.


## Description

A Fortran file `tensor_permutation.f90` is provided that demonstrates:

* constructing a Torch tensor from a Fortran array and inspecting its shape
  and strides;
* permuting tensor dimensions with the `permute_dims` argument (an involution
  / transpose), and the safeguard against invalid permutations;
* extracting permuted data back into a Fortran array with matching shape, and
  how permuting both into and out of Torch recovers the original array layout;
* demonstrating that tensor assignment copies values and shape but not strides
  (the output tensor keeps its own memory layout);
* comparing Fortran column-major and Torch row-major memory layout via printed
  output, showing the same data as it appears in memory, in Fortran, and in
  Torch.

## Dependencies

To run this example requires:

- CMake
- Fortran compiler
- FTorch (installed as described in main package)
- Python 3

## Running

To run this example, first install FTorch as described in the main
documentation, making use of the `examples` optional dependencies. See the
[user guide section](https://cambridge-iccs.github.io/FTorch/page/installation/general.html#python-dependencies)
on Python dependencies for details.

To compile the Fortran code, using the included `CMakeLists.txt`, execute the
following code:
```
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=<path/to/your/installation/of/library/> -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

(Note that the Fortran compiler can be chosen explicitly with the
`-DCMAKE_Fortran_COMPILER` flag, and should match the compiler that was used to
locally build FTorch.)

To run the compiled code, simply run the executable from the command line:
```
./tensor_permutation
```

You will also likely need to add the location of the dynamic library files
(`.so` or `.dylib` files) that we will link against at runtime to your
`LD_LIBRARY_PATH`:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:</path/to/library/installation>/lib
```
or `DYLD_LIBRARY_PATH` on Mac:
```
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:</path/to/library/installation>/lib
```
