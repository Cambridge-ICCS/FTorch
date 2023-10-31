title: When to transpose data

Transposition of data between Fortran and C can lead to a lot of unneccessary confusion.
The FTorch library looks after this for you with the
[`torch_tensor_from_array()`](doc/proc/torch_tensor_from_array.html) function which
allows you to index a tensor in Torch in **exactly the same way** as you would in Fortran.

If you wish to do something different to this then there are more complex functions
available and we describe here how and when to use them.

[TOC]

## Introduction - row- vs. column-major

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


## Why does this matter?

This matters for FTorch because a key feature is no-copy memory transfer between Fortran
and Torch.
To do this the Fortran data that will be used in Torch is stored in memory and a
[pointer](https://en.wikipedia.org/wiki/Pointer_(computer_programming)) to the first
element, \(a\) provided to Torch.

Now, if Torch were to take this block of memory and interpret it as as a 2x2 matrix it
would be read in as
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


## What can we do?

There are a few approaches we can take to address this.  
The first two of these are listed for conceptual purposes, whilst we advise handling
this in practice by using the `torch_tensor_from_array` function described in 
[4) below](#4-use-torch_tensor_from_array) or, if this is not suitable, the
the `layout` argument for `torch_tensor_from_blob` presented in
[3) below](#3-use-the-layout-argument-in-torch_tensor_from_blob).

#### 1) Transpose before passing
As seen from the above example, writing out from Fortran and reading directly in to
Torch results in us recieving the transpose.

Therefore we could transpose out Fortran data immediately before passing it to Torch.
As a result we will read in to Torch indexed the same as in Fortran pre-transposition.

For arrays of dimension 2 this can be done using the intrinsic
[`transpose()`](https://gcc.gnu.org/onlinedocs/gcc-12.1.0/gfortran/TRANSPOSE.html)
function.

For larger arrays we are required to use the
['reshape()'](https://gcc.gnu.org/onlinedocs/gfortran/RESHAPE.html) intrinsic to swap
the order of the indices.  
For example, if we had a 3x4x5 matrix \(A\) we would need to call
```
A_to_torch = reshape(A, shape=[5, 4, 3], order=[3, 2, 1])
```
which could then be read by Torch as a 3x4x5 tensor.

We would, of course, need to remember to transpose/reshape any output of the model
as required.

However, the transposition process involves creating a copy of the Fortran data.
For large matrices/tensors this can become expensive.
It would be better if we can pass data without having to transpose beforehand.

#### 2) Design nets to use transpose
Alternatively we could design our net to use
$$
\begin{pmatrix}
a & c \\
b & d
\end{pmatrix}
$$
as its input tensor meaning we can simply write from Fortran and read to Torch.

However, this requires foresight and may not be intuitive - we would like to be indexing
data in the same way in both Fortran and Torch.
Not doing so could leave us open to introducing bugs.

#### 3) Use the `layout` argument in `torch_tensor_from_blob`
In the [documentation for torch_tensor_from_blob](doc/proc/torch_tensor_from_blob.html)
there is a `layout` argument.

This tells us how data output by Fortran should be read/accessed by Torch.

TODO: Finish this
passing `layout = [1, 2]` means that the data will be read in the correct indices by
Torch.

Tells us which order to read the indices in.
i.e. `[1, 2]` will read `i` then `j`.

#### 4) Use `torch_tensor_from_array`

By far the easiest way to deal with the issue is not to worry about it at all!

As described at the top of this page, by using the
[torch_tensor_from_array](doc/proc/torch_tensor_from_array.html) function
you can take data from Fortran and send it to Torch to be indexed in exactly
the same way.

i.e. if the Fortran array `A`
$$
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
$$
is passed as `torch_tensor_from_array(A, )`
the resulting Tensor will be read by Torch as 




This is achieved by wrapping the `torch_tensor_from_blob` function to automatically
generate strides assuming this is what should happen.


## Advanced use

Those experienced with C will perhaps have noticed that there are further freedoms
available beyond those presented above.

[WIP]

Always stride, or is there a transpose tradeoff?


