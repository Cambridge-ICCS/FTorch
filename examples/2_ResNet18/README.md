# Example 2 - ResNet-18

This example provides a simple but complete demonstration of how to use the library.

## Description

A python file is provided that downloads the pretrained
[ResNet-18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
model from [TorchVision](https://pytorch.org/vision/stable/index.html).

A modified version of the `pt2ts.py` tool saves this ResNet-18 to TorchScript.

A series of files `resnet_infer_<LANG>` then bind from other languages to run the
TorchScript ResNet-18 model in inference mode.

## Dependencies

To run this example requires:

- cmake
- fortran compiler
- FTorch (installed as described in main package)
- python3

## Running

To run this example install fortran-pytorch-lib as described in the main documentation.
Then from this directory create a virtual environment an install the neccessary python
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

At this point we no longer require python, so can deactivate the virtual environment:
```
deactivate
```

To call the saved ResNet-18 model from fortran we need to compile the `resnet_infer`
files.
This can be done using the included `CMakeLists.txt` as follows:
```
mkdir build
cd build
cmake .. -DFTorch_DIR=<path/to/your/installation/of/library/>lib/cmake/ -DCMAKE_BUILD_TYPE=Release
make
```
Make sure that the  `FTorch_DIR` flag points to the `lib/cmake/` folder within the installation of the FTorch library.  

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


Alternatively we can use `make`, instead of cmake, with the included Makefile.
However, to do this you will need to modify `Makefile` to link to and include your
installation of FTorch as described in the main documentation. Also check that the compiler is the same as the one you built the Library with.  
You will also likely need to add the location of the `.so` files to your `LD_LIBRARY_PATH`:
```
make
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:</path/to/library/installation>/lib64
./resnet_infer_fortran saved_resnet18_model_cpu.pt
```

## Trying your own data

Alternative images and labels can be tested by modifying the following:

```python
import urllib

# Download a new image of a dog
url, filename = (
    "https://github.com/pytorch/vision/raw/v0.15.1/gallery/assets/dog1.jpg",
    "data/dog2.jpg",
)
urllib.request.urlretrieve(url, filename)

# Download ImageNet labels
url, filename = (
    "https://raw.githubusercontent.com/pytorch/hub/e55b003/imagenet_classes.txt",
    "imagenet_classes.txt",
)
data = urllib.request.urlopen(url)
categories = [s.strip().decode("utf-8") for s in data]
```

## Further options

To explore the functionalities of this model:

- Try saving the model through tracing rather than scripting by modifying `pt2ts.py`
