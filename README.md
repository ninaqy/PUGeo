# PUGeo-Net: A Geometry-centric Network for 3D Point Cloud Upsampling
- Paper: https://arxiv.org/abs/2002.10277

### Environment setting
The code is implemented with CUDA=10.0, tensorflow=1.14, python=2.7.
Other requested libraries: tqdm

### Training and testing data
Will release soon

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

### Training
python main.py --phase train --up_ratio 4 --log_dir PUGeo_x4

### Evaluate performance
python main.py --phase test --up_ratio 4 --pretrained PUGeo_x4/model/model-final --eval_xyz SketchFab/test
The upsampled xyz will be stored in PUGeo_x4/eval. The pretrained model will be released soon.

We thank the author of pointnet, pointnet2， PU-Net and MPU for their public code. 
