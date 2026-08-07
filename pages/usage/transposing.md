title: When to transpose data
author: Jack Atkinson
date: Last Updated: August 2026

## When to transpose data

Transposition of data between Fortran and C can lead to a lot of unnecessary confusion.
The FTorch library looks after this for you with the
[[ftorch_tensor(module):torch_tensor_from_array(interface)]] function which
allows you to index a tensor in Torch in **exactly the same way** as you would in Fortran.

If you wish to do something different to this then there are more complex functions
available and we describe here how and when to use them.
However, to be clear, we recommend sticking with the default behaviour unless you have
good reason not to. The defaults are likely to be efficient, are the least bug prone,
and cover the majority of user cases.


### Introduction - row- vs. column-major

Astute users will note that Fortran is a
[column-major](https://en.wikipedia.org/wiki/Row-_and_column-major_order)
language whilst C, C++, and Python are 
[row-major](https://en.wikipedia.org/wiki/Row-_and_column-major_order).

This means that the matrix/tensor in Fortran
$$
\begin{pmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{pmatrix}
=
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
$$
will appear in
[contiguous memory](https://en.wikipedia.org/wiki/Memory_management_(operating_systems))
on the computer as 
$$
\begin{pmatrix}
a_{11} & a_{21} & a_{12} & a_{22}
\end{pmatrix}
=
\begin{pmatrix}
a & c & b & d
\end{pmatrix}
$$
with the order of elements decided by moving down the columns before progressing in the
row dimension.  
In contrast, the same matrix/tensor defined in a row-major language will appear in
contiguous memory as
$$
\begin{pmatrix}
a_{11} & a_{12} & a_{21} & a_{22}
\end{pmatrix}
=
\begin{pmatrix}
a & b & c & d
\end{pmatrix}
$$
reading along each row before progressing down the column dimension.


### Why does this matter?

This matters for FTorch because a key feature is no-copy memory transfer between Fortran
and Torch.
To do this the Fortran data that will be used in Torch is stored in memory and a
[pointer](https://en.wikipedia.org/wiki/Pointer_(computer_programming)) to the first
element, \(a\) provided to Torch.

Now, if Torch were to take this block of memory and na&iuml;vely interpret it as a
2x2 matrix it would be read in as
$$
\begin{pmatrix}
a & c \\
b & d
\end{pmatrix}
$$
which is the [transpose](https://en.wikipedia.org/wiki/Transpose) of the
matrix we had in Fortran; likely not what we were expecting!

This means we need to be careful when passing data to make sure that what we read in
to our Torch net is correct as we expect.

FTorch handles this by constructing all tensors associated with Fortran data using
column-major strides. This means that data can be transferred without any copying or
rearranging, and indexed in exactly the same way in both Fortran and Torch - the most
predictable behaviour, with the fewest opportunities for bugs.


### When might you want the transpose?

The price of FTorch matching the indexing is that tensor data is strided (non-contiguous)
from Torch's row-major perspective.
This has the potential to increase cache-misses when accessing the data.
Torch handles strided tensors well, and this penalty is often smaller than the cost
of rearranging the data, so the default behaviour in FTorch is the best choice for most users.

Sometimes you may genuinely want row-major (contiguous) data in Torch however.
For example if profiling shows data access to be a bottleneck, or to interoperate
with nets that expect it.

The only way to achieve this (without copying or re-arranging memory) is for Torch to
interpret the data as the transpose of the Fortran array:
the memory \(a, c, b, d\) read contiguously in row-major order is
$$
A^T =
\begin{pmatrix}
a & c \\
b & d
\end{pmatrix}
$$
The options below describe the ways to do this and their associated costs.


### What are the options?

#### 1) Transpose in Fortran before passing

The most direct way to control the memory layout is to rearrange the data
in Fortran before passing it to Torch.

For rank-2 arrays this can be done using the intrinsic
[`transpose()`](https://gcc.gnu.org/onlinedocs/gcc-12.1.0/gfortran/TRANSPOSE.html)
function.

For higher ranks use the
['reshape()'](https://gcc.gnu.org/onlinedocs/gfortran/RESHAPE.html) intrinsic to swap
the order of the indices.
For example, if we had a 3x4x5 array \(A\) we would call
```
A_to_torch = reshape(A, shape=[5, 4, 3], order=[3, 2, 1])
```
Passing the result with `permute_dims=[n, ..., 1]` (see option 2 below) then gives
a Torch tensor that is indexed exactly like the original Fortran array _and_ is
contiguous row-major in memory - the best of both worlds.

However, transposition copies all of the data which for large arrays is expensive.
Doing this will likely cost more than the cache-misses incurred by FTorch's default
strided access.
This also requires permuting and transposing to get the same conceptual indexing result
as the default behaviour, increasing the potential for bugs.

#### 2) Use `permute_dims` to pass the transpose

Alternatively, we could design our net so that its input is the transpose of our
Fortran data
$$
\begin{pmatrix}
a & c \\
b & d
\end{pmatrix}
$$
Reading the shared memory contiguously with Torch's row-major strides then gives the net
what it expects with no data copying required.

The practical way to achieve this is to use the optional `permute_dims` argument of
[[ftorch_tensor(module):torch_tensor_from_array(interface)]].
This takes an array specifying a permutation of the dimensions and matches the
semantics of PyTorch's
[`torch.permute()`](https://docs.pytorch.org/docs/stable/generated/torch.permute.html)
with element `i` indicating which dimension of the Fortran array becomes dimension `i` of
the Torch tensor.
FTorch permutes the shape and strides of the tensor accordingly, so passing
`permute_dims=[n, ..., 1]` exposes the data contiguously as the transpose of the Fortran array.

i.e. if the Fortran array `A`
$$
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
$$
is passed as `torch_tensor_from_array(tensor, A, torch_device, permute_dims=[2, 1])`
the resulting tensor will be indexed by Torch as
$$
\begin{pmatrix}
a & c \\
b & d
\end{pmatrix}
$$
with contiguous strides `[2, 1]`.

Torch code therefore needs to work with the transposed indexing compared to Fortran.
This requires foresight to design code with the expectation of coupling, and indexing
differently between your Fortran and Torch code can be a source of bugs, so beware.

@Note
Permutations beyond axis-reversal, e.g. a cyclic `[3, 1, 2]`, are also possible.
These produce tensors that are neither row- nor column-major strided, but can be
useful for remapping data between conventions - for example moving a channel
dimension from first to last position.
@endnote

@note
For a detailed exercise demonstrating `permute_dims` and its effect on shape, strides,
and memory layout, see the
[tensor permutation worked example](https://github.com/Cambridge-ICCS/FTorch/tree/main/examples/12_Permutation).
@endnote


### Advanced use with `torch_tensor_from_blob`

For more advanced options for manipulating and controlling data access when passing
between Fortran and Torch see the more powerful but more complex
[[ftorch_tensor(module):torch_tensor_from_blob(subroutine)]] subroutine which allows
users to directly specify the shape and strides with which to construct tensors.
