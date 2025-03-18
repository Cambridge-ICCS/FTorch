# Example 2 - ResNet-18

This example provides a more realistic demonstration of how to use the library, using ResNet-18 to classify an image.

As the input to this model is four-dimensional (batch size, colour, x, y), care must be taken dealing with the data array in Python and Fortran. See [When to transpose arrays](#when-to-transpose-arrays) for more details.

## Description

A Python file is provided that downloads the pretrained
[ResNet-18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
model from [TorchVision](https://pytorch.org/vision/stable/index.html).

A modified version of the `pt2ts.py` tool saves this ResNet-18 to TorchScript.

A series of files `resnet_infer_<LANG>` then bind from other languages to run the
TorchScript ResNet-18 model in inference mode.

## Dependencies

To run this example requires:

- CMake
- Fortran compiler
- FTorch (installed as described in main package)
- Python 3

## Running

To run this example install FTorch as described in the main documentation.
Then from this directory create a virtual environment an install the neccessary Python
modules:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You can check that everything is working by running `resnet18.py`:

```
python3 resnet18.py
```

When using single precision, it should produce the result:

```
Top 5 results:

Samoyed (id=258): probability = 0.8846225142478943
Arctic fox (id=279): probability = 0.045805174857378006
white wolf (id=270): probability = 0.0442761555314064
Pomeranian (id=259): probability = 0.005621383432298899
Great Pyrenees (id=257): probability = 0.004652013536542654
```

To save the pretrained ResNet-18 model to TorchScript run the modified version of the
`pt2ts.py` tool :
```
python3 pt2ts.py
```

At this point we no longer require Python, so can deactivate the virtual environment:
```
deactivate
```

To call the saved ResNet-18 model from Fortran we need to compile the `resnet_infer`
files.
This can be done using the included `CMakeLists.txt` as follows:
```
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=<path/to/your/installation/of/library/> -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

(Note that the Fortran compiler can be chosen explicitly with the `-DCMAKE_Fortran_COMPILER` flag,
and should match the compiler that was used to locally build FTorch.)

To run the compiled code calling the saved ResNet-18 TorchScript from Fortran run the
executable with an argument of the saved model file:
```
./resnet_infer_fortran ../saved_resnet18_model_cpu.pt
```

This should produce the same top result:

```
 Top result

 Samoyed (id=         259 ), : probability =  0.884623706
```


Alternatively we can use `make`, instead of CMake, with the included Makefile.
However, to do this you will need to modify `Makefile` to link to and include your
installation of FTorch as described in the main documentation. Also check that the compiler is the same as the one you built the Library with.  
You will also likely need to add the location of the `.so` files to your `LD_LIBRARY_PATH`:
```
make
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:</path/to/library/installation>/lib
./resnet_infer_fortran saved_resnet18_model_cpu.pt
```

## Further options

To explore the functionalities of this model:

- Try saving the model through tracing rather than scripting by modifying `pt2ts.py`
- Try changing the input image. For example, running the following code will download an alternative image of a dog from [pytorch/vision](https://github.com/pytorch/vision/tree/v0.15.2/gallery/assets), saving it in the `data` directory:

```python
import urllib

url, filename = (
    "https://github.com/pytorch/vision/raw/v0.15.1/gallery/assets/dog1.jpg",
    "data/dog2.jpg",
)
urllib.request.urlretrieve(url, filename)
```

`image_filename` in resnet18.py and resnet_infer_python.py must then be modified to match the name of this new file.

`expected_prob` in resnet_infer_fortran.f90 and resnet_infer_python.py are also specific to the pre-downloaded example image, so either their values should be updated, or the assertions should be removed.


Note that the ImageNet labels can be downloaded and accessed similarly, using:

```python
import urllib

url, filename = (
    "https://raw.githubusercontent.com/pytorch/hub/e55b003/imagenet_classes.txt",
    "imagenet_classes.txt",
)
data = urllib.request.urlopen(url)
categories = [s.strip().decode("utf-8") for s in data]
```

## When to transpose arrays?

In this example, it is expected that the shape and indices of `in_data` in resnet_infer_fortran.f90 match that of `input_batch` in resnet18.py, i.e. `in_data(i, j, k, l) ==  input_batch[i, j, k, l]`.

Since C is row-major (rows are contiguous in memory), whereas Fortran is column-major (columns are contiguous), it is therefore necessary to perform a transpose when converting from the NumPy array to the Fortran array to ensure that their indices are consistent.

In this example code, the NumPy array is transposed before being flattened and saved to binary, allowing Fortran to `reshape` the flatted array into the correct order.

An alternative would be to save the NumPy array with its original shape, but perform a transpose during or after reading the data into Fortran, e.g. using:

```
in_data = reshape(flat_data, shape(in_data), order=(4,3,2,1))
```

For more general use, it should be noted that the function used to create the input tensor from `input_batch`, `torch_tensor_from_blob`, performs a further transpose, which is required to allow the tensor to interact correctly with the model.
