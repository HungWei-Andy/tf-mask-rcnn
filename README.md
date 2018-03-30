# Tensorflow Mask-RCNN
This is an tensorflow implemetation of [Kaming He, et al. Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf). The paper reports two backbone network features: [ResNet50](https://arxiv.org/pdf/1512.03385.pdf) + FPN and ResNet50 + C4 features. This implementation utilizes [Mobilenet v1](https://arxiv.org/pdf/1704.04861.pdf) with 0.5 width multiplier + FPN and ResNet50 + C5 as backbone. The implementations is trined and tested using COCO 2014 train and validation dataset. 

## Download Code
Since the code contains a cocoapi submodule, be noticed to specify --recursive for git clone to download.
```bash
git clone --recursive https://github.com/HungWei-Andy/tf-mask-rcnn.git
```

## COCO Dataset
For training and testing on COCO dataset, please download corresponding dataset from [COCO official website](http://cocodataset.org/#download). After downloading the dataset, put the annotations into ./COCO/annotations and images into ./COCO/images. For examples,
```
./COCO/annotations/instances_train2014.json
./COCO/annotations/instances_val2014.json
./COCO/images/train2014/xxxxxx.jpg
./COCO/iamges/val2014/xxxxxx.jpg
``` 
The format follows the [official demo code inside coco python api](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb).

After the coco dataset is downloaded, goto ./lib/loader/cocoapi/PythonAPI and run make to compile the COCO Python API.
```bash
cd ./lib/loader/cocoapi/PythonAPI
make
```

## Requirements
The required python libraries is listed in requirements.txt, install these packages before running the code.
```bash
pip install -r requirements.txt
```

## Pretrained Model
The imagenet pretrained weight of resnet50 and mobilenet v1 can be downloaded in [Google Drive](https://drive.google.com/file/d/18oSnUcrs8YWKTWmRJMf3Tm2r12fNJLDJ/view?usp=sharing) and [Baidu Cloud](https://pan.baidu.com/s/1qsy76EQ-Jc1OqEeKv4IBOA).

For ResNet50, since the tensorflow slim model of residual network has many issues and is different from the original one in paper [https://github.com/tensorflow/models/issues/2130](https://github.com/tensorflow/models/issues/2130), I hand-crafted a residual network model in tensorflow including ResNet50, ResNet101, ResNet152 according to [the official caffe deploy document](https://github.com/KaimingHe/deep-residual-networks#models). I converted the caffemodel weight using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) by ethereon. The names of corresponding variables are matched to the deploy document. A code to load .npy file for variable tensors is also provided. The details can be refered to "./lib/nets/resnet.py"

For mobilenet v1, the slim model is adopted including the code and the pretrained weights[link](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md). For more details, mobilenet v1 with 0.5 width multiplier and input size 224 without quantization is adopted (MobileNet_v1_0.50_224).

## Training Models
The ResNet50 + C5 backbone is trained for 86000 iterations with batch size 2 using momentum optimization. The mobilenet v1 + FPN backbone is trained for 84000 iterations using adam optimization with learning rate 1e-4. Other settings follows the original work. By the way, the mobilenet v1 backbone is parallel trained using momentum and adam. When adam shows faster convergence in around 25000 iterations, we shut down the training using momentum. The trained model can be downloaded from [Google Drive](https://drive.google.com/file/d/1U_rTjPQKLDKiUsgqekO0wR5lZCiJ4uSZ/view?usp=sharing) and [Baidu Cloud](https://pan.baidu.com/s/1WWSX5IQFUjG-ij-68u65SA). These weights includes the network weights and the optimization parameters.

## Training and Testing
To run the training and testing on COCO dataset, use "train.sh" and "test.sh". Check the configuration in ./lib/config.py before running the code. The loaded weights is the latest model in cfg.output_dir directory.

