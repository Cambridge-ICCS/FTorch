# Example 1 - Tensor manipulation

This example provides a simple demonstration of how to create, manipulate,
interrogate, and destroy instances of the `torch_tensor` derived type. This is
one of the core derived types in the FTorch library, providing an interface to
the `torch::Tensor` C++ class. Like `torch::Tensor`, the `torch_tensor` derived
type is designed to have a similar API to PyTorch's `torch.Tensor` class.


## Description

A Fortran file `tensor_manipulation.f90` is provided that demonstrates handling
of the `torch_tensor` derived type.

## Dependencies

To run this example requires:

- CMake
- Fortran compiler
- FTorch (installed as described in main package)
- Python 3

## Running

To run this example, first install FTorch as described in the main
documentation.

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
./tensor_manipulation
```

Alternatively we can use `make`, instead of CMake, with the included Makefile.
However, to do this you will need to modify `Makefile` to link to and include
your installation of FTorch as described in the main documentation. Also check
that the compiler is the same as the one you built the Library with.
```
make
./tensor_manipulation
```

You will also likely need to add the location of the dynamic library files
(`.so` or `.dylib` files) that we will link against at runtime to your
`LD_LIBRARY_PATH`:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:</path/to/library/installation>/lib
```
or `DYLD_LIBRARY_PATH` on mac:  
```
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:</path/to/library/installation>/lib
```
