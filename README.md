# Mask TextSpotter

### A Pytorch implementation of Mask TextSpotter along with its extension can be find [here](https://github.com/MhLiao/MaskTextSpotter)

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
4. [Datasets](#datasets)
5. [Test](#test)
6. [Train](#train)

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
Download the model and place it as ```models/model_iter79999.pkl```
Our trained model:
[Google Drive](https://drive.google.com/open?id=1yPATzUCREBopDIHcsvdYOBB3YpStunMU);
[BaiduYun](https://pan.baidu.com/s/1JPZmOQ1LAw98s0GPa-PuuQ) (key of BaiduYun: gnpc)

### Datasets
Download the ICDAR2013([Google Drive](https://drive.google.com/open?id=1sptDnAomQHFVZbjvnWt2uBvyeJ-gEl-A), [BaiduYun](https://pan.baidu.com/s/18W2aFe_qOH8YQUDg4OMZdw)) and ICDAR2015([Google Drive](https://drive.google.com/open?id=1HZ4Pbx6TM9cXO3gDyV04A4Gn9fTf2b5X), [BaiduYun](https://pan.baidu.com/s/16GzPPzC5kXpdgOB_76A3cA)) as examples.
Datasets should be placed in ```lib/datasets/data/``` as below
```
synth
icdar2013
icdar2015
scut-eng-char
totaltext
```
If you do not train the model, you can just download the ICDAR2013 or ICDAR2015 datasets for testing.

### Test
```
python tools/test_net.py --cfg configs/text/mask_textspotter.yaml
```
You can modify the model path or the test dataset in ```configs/text/mask_textspotter.yaml```.

### Train
You should format all the datasets you used for training as above.
Then modify ```configs/text/mask_textspotter.yaml``` to fit the gpus, model path, and datasets.
```
python tools/train_net.py --cfg configs/text/mask_textspotter.yaml
```

