# Mask TextSpotter

### Introduction
This is the official implementation of Mask TextSpotter.

Mask TextSpotter is an End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes

For more details, please refer to our [paper](https://arxiv.org/abs/1807.02242). 

### Citing the paper

Please cite the paper in your publications if it helps your research:

    @inproceedings{LyuLYWB18,
      author    = {Pengyuan Lyu and
                   Minghui Liao and
                   Cong Yao and
                   Wenhao Wu and
                   Xiang Bai},
      title     = {Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes},
      booktitle = {Proc. ECCV},
      pages     = {71--88},
      year      = {2018}
    }
    
   

### Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Models](#models)
4. [Test](#test)
5. [Train](#train)

### Requirements
- NVIDIA GPU, Linux, Python2
- Caffe2, various standard Python packages

### Installation
#### Caffe2

To install Caffe2 with CUDA support, follow the [installation instructions](https://caffe2.ai/docs/getting-started.html) from the [Caffe2 website](https://caffe2.ai/). **If you already have Caffe2 installed, make sure to update your Caffe2 to a version that includes the [Detectron module](https://github.com/caffe2/caffe2/tree/master/modules/detectron).**

Please ensure that your Caffe2 installation was successful before proceeding by running the following commands and checking their output as directed in the comments.

```
# To check if Caffe2 build was successful
python2 -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

# To check if Caffe2 GPU build was successful
# This must print a number > 0 in order to use Detectron
python2 -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
```

If the `caffe2` Python package is not found, you likely need to adjust your `PYTHONPATH` environment variable to include its location (`/path/to/caffe2/build`, where `build` is the Caffe2 CMake build directory).

Install Python dependencies:

```
pip install numpy pyyaml matplotlib opencv-python>=3.0 setuptools Cython mock
```

Set up Python modules:

```
cd $ROOT_DIR/lib && make
```

Note: Caffe2 is difficult to install sometimes.


### Models

Our trained model:
[Google Drive](https://drive.google.com/open?id=1yPATzUCREBopDIHcsvdYOBB3YpStunMU);
[BaiduYun](链接：https://pan.baidu.com/s/1JPZmOQ1LAw98s0GPa-PuuQ) (key of BaiduYun: gnpc)

### Test

### Train

