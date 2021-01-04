# PUGeo-Net: A Geometry-centric Network for 3D Point Cloud Upsampling
This is the official Tensorflow implementation for paper: https://arxiv.org/abs/2002.10277

### Environment setting
The code is implemented with CUDA=10.0, tensorflow=1.14, python=2.7. Other settings should also be ok.

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
One should change the CUDA path in compile.sh.

Some common problems during compiling:
- Make sure you change the CUDA path in compile.sh correctly.
- Make sure you are using (and also compile under) tensorflow-gpu, not the cpu version of TF.
- You may compile with other TF version. May need to modify the compile.sh. One can refer to the issues of pointnet2, PU-Net, MPU and PU-GAN.
- Check the "libtensorflow_framework.so" in your tensorflow folder, if it is installed as "libtensorflow_framework.so.1", run this command:
```
ln -s  libtensorflow_framework.so.1  libtensorflow_framework.so
```

For any question regarding to the compiling, one can refer to the issues of pointnet2, PU-Net, MPU and PU-GAN.

### Datasets and pretrained model
We provide x4 training dataset and pretrained model. Please download these files in the following link:
- training data (tfrecord_x4_normal.zip)
- 13 testing models with 5000 points (test_5000.zip) 
- pretrained x4 model (PUGeo_x4.zip) 

https://drive.google.com/drive/folders/1n2lf4am9k3hy3ci4W20XiMkXwJKwyg8f?usp=sharing

### Training
```
python main.py --phase train --up_ratio 4 --log_dir PUGeo_x4
```

### Inference (upsampling)
```
python main.py --phase test --up_ratio 4 --pretrained PUGeo_x4/model/model-final --eval_xyz test_5000
```
The upsampled xyz will be stored in PUGeo_x4/eval.

We thank the authors of pointnet2 PU-Net and MPU for their public code. 
