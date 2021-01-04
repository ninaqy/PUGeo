# PUGeo-Net: A Geometry-centric Network for 3D Point Cloud Upsampling
- Paper: https://arxiv.org/abs/2002.10277

### Environment setting
The code is implemented with CUDA=10.0, tensorflow=1.14, python=2.7.
Other requested libraries: tqdm

### Compile tf_ops
```
cd tf_ops/CD
sh compile.sh
```
```
cd  tf_ops/sampling
sh compile.sh
```
For any question regarding the compiling problem, one can refer to the issues of pointnet2, PU-Net, MPU and PU-GAN

### Datasets and pretrained model
We provide x4 training dataset and pretrained model. Please download training data (tfrecord_x4_normal.zip), 13 testing models with 5000 points (test_5000.zip) and pretrained x4 model (PUGeo_x4.zip) in the following link:
https://drive.google.com/drive/folders/1n2lf4am9k3hy3ci4W20XiMkXwJKwyg8f?usp=sharing

### Training
python main.py --phase train --up_ratio 4 --log_dir PUGeo_x4

### Evaluate performance
python main.py --phase test --up_ratio 4 --pretrained PUGeo_x4/model/model-final --eval_xyz SketchFab/test
The upsampled xyz will be stored in PUGeo_x4/eval.

We thank the author of pointnet, pointnet2， PU-Net and MPU for their public code. 
